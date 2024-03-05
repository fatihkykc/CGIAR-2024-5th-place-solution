import os

# from ffcv.fields.ndarray import NDArrayDecoder

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-12.3/bin/ptxas"
# os.environ["LD_LIBRARY_PATH"] += "/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so"
import warnings

import numpy as np
from fastai import *
from fastai.vision.all import *
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
from functools import partial

from fastai.callback.wandb import *
from fastai.vision.all import Learner
from fastxtend.callback.compiler import CompilerCallback
from fastxtend.ffcv.all import *
from fastxtend.vision.all import *
from sklearn.model_selection import StratifiedKFold
from torch import nn

# import wandb

# os.system("wandb login 5e93b528a6f7fdd00749d7fa1fa5545d489ecb2f")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True


# %%
# Override the after epoch of tracker callback, to ensure that we can start from late epochs, to continue from already trained models
def after_epoch2(self):
    "Compare the last value to the best up to now"
    try:
        val = self.recorder.values[-1][self.idx]
    except:
        val = np.inf
    if self.comp(val - self.min_delta, self.best):
        self.best, self.new_best = val, True
    else:
        self.new_best = False


TrackerCallback.after_epoch = after_epoch2


# Override the earlystopping callback, to stall the earlystopping count until the start_epoch
class EarlyStoppingCallback2(TrackerCallback):
    "A `TrackerCallback` that terminates training when monitored quantity stops improving."
    order = TrackerCallback.order + 3

    def __init__(
        self,
        monitor="valid_loss",  # value (usually loss or metric) being monitored.
        comp=None,  # numpy comparison operator; np.less if monitor is loss, np.greater if monitor is metric.
        min_delta=0.0,  # minimum delta between the last monitor value and the best monitor value.
        patience=1,  # number of epochs to wait when training has not improved model.
        reset_on_fit=True,  # before model fitting, reset value being monitored to -infinity (if monitor is metric) or +infinity (if monitor is loss).
    ):
        super().__init__(
            monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit
        )
        self.patience = patience

    def before_fit(self):
        self.wait = 0
        super().before_fit()

    def after_epoch(self):
        "Compare the value monitored to its best score and maybe stop training."
        if self.learn.ep_start <= self.epoch:
            super().after_epoch()
            if self.new_best:
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(
                        f"No improvement since epoch {self.epoch-self.wait}: early stopping"
                    )
                    raise CancelFitException()


# Override the earlystopping callback, to stall the earlystopping count until the start_epoch
class StopAtEpochCallback(Callback):
    "A `TrackerCallback` that terminates training when monitored quantity stops improving."

    def __init__(
        self,
        max_epochs=None,  # number of epochs to wait when training has not improved model.
        reset_on_fit=True,  # before model fitting, reset value being monitored to -infinity (if monitor is metric) or +infinity (if monitor is loss).
    ):
        self.max_epochs = max_epochs

    def after_epoch(self):
        "Compare the value monitored to its best score and maybe stop training."
        if self.max_epochs == self.epoch:
            print(f"Stopping at epoch {self.max_epochs}")
            raise CancelFitException()


# Override the save method of the learner, to add the EMA model to save.
def save2(self: Learner, file, **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    try:
        obj = next((x for x in self.cbs if isinstance(x, EMACallback)), None)
        if obj is not None:
            file_ema = join_path_file(
                file + "_ema", self.path / self.model_dir, ext=".pth"
            )
            save_model(file_ema, obj.ema_model, getattr(self, "opt", None), **kwargs)
            self.best_model_ema = deepcopy(obj.ema_model)
            print("saved best ema model")
    except:
        pass
    file = join_path_file(file, self.path / self.model_dir, ext=".pth")
    save_model(file, self.model, getattr(self, "opt", None), **kwargs)
    return file


class SaveModelCallback2(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    order = TrackerCallback.order + 1

    def __init__(
        self,
        monitor="valid_loss",  # value (usually loss or metric) being monitored.
        comp=None,  # numpy comparison operator; np.less if monitor is loss, np.greater if monitor is metric.
        min_delta=0.0,  # minimum delta between the last monitor value and the best monitor value.
        fname="model",  # model name to be used when saving model.
        every_epoch=False,  # if true, save model after every epoch; else save only when model is better than existing best.
        at_end=False,  # if true, save model when training ends; else load best model if there is only one saved model.
        with_opt=False,  # if true, save optimizer state (if any available) when saving model.
        reset_on_fit=True,  # before model fitting, reset value being monitored to -infinity (if monitor is metric) or +infinity (if monitor is loss).
    ):
        super().__init__(
            monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit
        )
        assert not (
            every_epoch and at_end
        ), "every_epoch and at_end cannot both be set to True"
        # keep track of file path for loggers
        self.last_saved_path = None
        store_attr("fname,every_epoch,at_end,with_opt")

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            if (self.epoch % self.every_epoch) == 0:
                self._save(f"{self.fname}_{self.epoch}")
        else:  # every improvement
            super().after_epoch()
            if self.new_best:
                print(
                    f"Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}."
                )
                # self.fname=f'{self.fname}_{self.best:.3f}'
                self._save(f"{self.fname}")

    # save model callback loads the model after fit, this causes to load the non-ema model after the freeze train step on finetune method.
    def after_fit2(self, **kwargs):
        "Load the best model."
        if self.at_end:
            self._save(f"{self.fname}")
        elif not self.every_epoch:
            self.learn.load(
                f"{self.fname}_ema" if self.ema_model else f"{self.fname}",
                with_opt=self.with_opt,
            )


"""
TODO: Do something with same pictures between train and test
"""


# %%
class CFG:
    arc = "vit_base_patch16_clip_384"
    loss_fn = "CE"
    optimizer = "AdamW"
    lr = 1e-3
    image_size = 384
    resize_method = "squish"
    batch_size = 82
    epochs = 17
    max_epochs = 17
    continue_from = False
    DEBUG = True
    freeze_epochs = 1
    tta = 8
    early_stop = 10
    label_smoothing = 0.005
    reduce_lr_delta = False
    wandb = False
    SEED = 0
    EMA = False
    fit_method = "one_cycle"
    soft_labels = False
    start_epoch = 0
    num_worker = 12
    infer = False
    compile = True
    channels_last = False
    cv = False
    weighted_CE = False
    weighted_dl = False
    relabeled = False
    train_smaller = False
    pseudo_labels = False
    neglect_low_sample_classes = False
    progressive_resize = False
    ffcv = False
    id = "first"


class AUG:
    cutmix = False
    mixup = False
    aug_tfms = True
    random_resized_crop_gpu = True
    random_erasing_batch = False
    normalize = True


if CFG.DEBUG:
    CFG.freeze_epochs = 1
    CFG.epochs = 1
    CFG.compile = False
    CFG.tta = 2
    CFG.early_stop = 2
    # CFG.arc = "vit_small_patch16_224"
    # CFG.image_size = 224

# %%
cfg_dict = {
    attr: getattr(CFG, attr)
    for attr in dir(CFG)
    if not callable(getattr(CFG, attr)) and not attr.startswith("__")
}
aug_dict = {
    attr: getattr(AUG, attr)
    for attr in dir(AUG)
    if not callable(getattr(AUG, attr)) and not attr.startswith("__")
}
config = {"CFG": cfg_dict, "AUG": aug_dict}

id = CFG.id

# id="evalarge"
if CFG.continue_from and not CFG.infer:
    os.makedirs(f"models/{CFG.arc}/{CFG.continue_from}/{id}", exist_ok=True)
elif not CFG.infer:
    os.makedirs(f"models/{CFG.arc}/{id}", exist_ok=True)

set_seed(CFG.SEED, reproducible=True)

# %%
TRAIN = (
    "image_features/changed_labels.csv"
    if CFG.relabeled
    else (
        "Train2.csv"
        if not CFG.soft_labels
        else f"models/{CFG.soft_labels}/Train_soft_0.7.csv"
    )
)
TEST = "Test2.csv"
# VAL = "Test.csv" if not CFG.soft_labels else "test_soft.csv"
print(TRAIN)
if CFG.DEBUG:
    Train = pd.read_csv(TRAIN, nrows=100)
    Test = pd.read_csv(TEST, nrows=100)
else:
    Train = pd.read_csv(TRAIN)
    Test = pd.read_csv(TEST)
Train = Train[Train["is_test"] == 0] if CFG.relabeled else Train
# Train.drop(["is_test","fname","leak_fname"], axis=1, inplace=True)


vocab = Train.label_id.unique()
print(vocab)
if CFG.cv:
    skf = StratifiedKFold(n_splits=CFG.cv, shuffle=True, random_state=CFG.SEED)
    Train["fold"] = -1
    for i, (train_idx, val_idx) in enumerate(skf.split(Train, Train["label_id"])):
        Train.loc[val_idx, "fold"] = i
    Train_df = Train
else:
    if CFG.train_smaller:
        Train["fold"] = -1
        skf = StratifiedKFold(
            n_splits=CFG.train_smaller, shuffle=True, random_state=CFG.SEED
        )
        for i, (train_idx, val_idx) in enumerate(skf.split(Train, Train["label_id"])):
            Train.loc[val_idx, "fold"] = i
            break
        Train = Train[Train["fold"] == 0]
    # val["is_valid"] = True
    Train["is_valid"] = False
    Train_df = Train
    # Train_df = pd.concat([Train, val]).reset_index(drop=True)
    # Test_df = Train[Train["is_valid"] == True]

import numpy as np

if CFG.pseudo_labels:
    PSEUDO = glob.glob(f"models/{CFG.pseudo_labels}/submission_*.csv")
    # PSEUDO=["val_pseudo_labels_0.527012831.csv"]
    print(PSEUDO)
    PSEUDO = PSEUDO[0]
    pseudo = pd.read_csv(PSEUDO)
    pseudo["is_valid"] = False
    pseudo["label_id"] = val["label_id"]
    pseudo["path"] = val["path"]
    if CFG.soft_labels:
        pseudo.rename(
            columns={
                "WD": "DR_soft",
                "G": "G_soft",
                "DR": "ND_soft",
                "ND": "WD_soft",
                "other": "other_soft",
            },
            inplace=True,
        )
    else:
        pseudo["label_id"] = np.argmax(pseudo[vocab].to_numpy(), axis=1)
        class_map = ["DR", "G", "ND", "WD", "other"]
        pseudo["label_id"] = pseudo["label_id"].apply(lambda x: class_map[x])

    # Train_df = Train_df[['ID']]
    print(pseudo.iloc[-1])
    # Train_df = Train_df.join(Train_df, pseudo, on="ID")
    Train_df = pd.concat([Train_df, pseudo]).reset_index(drop=True)

if CFG.neglect_low_sample_classes:
    Train_df.loc[Train_df["label_id"].isin(["OTHER", "ND"]), "label_id"] = "other"
    val.loc[val["label_id"].isin(["OTHER", "ND"]), "label_id"] = "other"


def splitter(df):
    train = df.index[df["is_valid"] == False].tolist()
    val = df.index[df["is_valid"] == True].tolist()
    return train, val


# %%
from sklearn.utils.class_weight import compute_class_weight

if CFG.weighted_CE:
    class_weights = compute_class_weight(
        class_weight="balanced", classes=vocab, y=Train_df.label_id
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

# %%


class FocalLossLBS(Module):
    y_int = True  # y interpolation

    def __init__(
        self,
        gamma: float = 2.0,  # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        weight: Tensor = None,  # Manual rescaling weight given to each class
        reduction: str = "mean",  # PyTorch reduction to apply to the output
        label_smoothing: int = 0,
    ):
        "Applies Focal Loss: https://arxiv.org/pdf/1708.02002.pdf"
        store_attr()

    def forward(self, inp: Tensor, targ: Tensor) -> Tensor:
        "Applies focal loss based on https://arxiv.org/pdf/1708.02002.pdf"
        ce_loss = F.cross_entropy(
            inp,
            targ,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def get_loss_fn(loss_name):
    if loss_name == "Focal":
        loss = FocalLossLBS(label_smoothing=CFG.label_smoothing, gamma=1.6)
    if loss_name == "CE":
        loss = nn.CrossEntropyLoss(
            label_smoothing=CFG.label_smoothing,
            weight=class_weights if CFG.weighted_CE else None,
        )
    if loss_name == "LBCE":
        loss = LabelSmoothingCrossEntropy()
    if loss_name == "BCE":
        loss = BCEWithLogitsLossFlat()
    return loss


# %%
def get_optimizer(opt):
    optimizer = None
    if opt == "lion":
        optimizer = lion(foreach=True)
    if opt == "Adam":
        t_adam = partial(OptimWrapper, opt=torch.optim.Adam)
        optimizer = partial(t_adam, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    elif opt == "AdamW":
        optimizer = adam(foreach=True)
    elif opt == "SGD":
        optimizer = partial(torch.optim.SGD)
    elif opt == "RMSprop":
        optimizer = partial(torch.optim.RMSprop, eps=1e-08)
    return optimizer


from fastai.data.all import *
from fastai.learner import *
from fastai.optimizer import *

mk_class(
    "ActivationType",
    **{o: o.lower() for o in ["No", "Sigmoid", "Softmax", "BinarySoftmax"]},
    doc="All possible activation classes for `AccumMetric",
)
logloss = skm_to_fastai(
    log_loss,
    labels=[0, 1, 2, 3, 4],
    is_class=False,
    axis=1,
    flatten=False,
    activation=ActivationType.Softmax,
)

mk_class(
    "ActivationType",
    **{o: o.lower() for o in ["No", "Sigmoid", "Softmax", "BinarySoftmax"]},
    doc="All possible activation classes for `AccumMetric",
)
logloss_multi = skm_to_fastai(
    log_loss, is_class=False, axis=1, flatten=False, activation=ActivationType.Softmax
)


def cross_entropy(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


logloss_ml = F.cross_entropy

# %%
root_dir = f"models/{CFG.arc}/{id}"
if CFG.soft_labels:
    vocab = ["DR_soft", "G_soft", "ND_soft", "WD_soft", "other_soft"]


# %%
def log_infer(preds, ema=False, fold="", write=False):
    # class_map = 'class_map.txt'

    # with open(class_map, 'r') as f:
    #     class_map = f.read()
    # class_map = class_map.split('\n')
    class_map = (
        ["DR", "G", "WD", "other"]
        if CFG.neglect_low_sample_classes
        else ["DR", "G", "ND", "WD", "other"]
    )
    # for i, cls in enumerate(class_map):
    #     Test.loc[Test["label_id"] == cls, "label_id"] = i

    # labelz = val["label_id"].tolist()
    # labels = [0, 1, 2, 3] if CFG.neglect_low_sample_classes else [0, 1, 2, 3, 4]
    # logloss = log_loss(y_true=labelz, y_pred=preds, labels=labels)
    # print("logloss test:", logloss)
    # acc = accuracy(preds, torch.tensor(labelz))
    # print("acc test: ", acc)

    # calc logloss for each class
    # np_labels = Test["label_id"].to_numpy()
    # losses = []
    # if write:
    #     if not os.path.exists(f"{root_dir}"):
    #         os.makedirs(f"{root_dir}")
    #     with open(f"{root_dir}/class_losses{ema}.txt", "w") as f:
    #         for i in range(len(vocab)):
    #             idxs = np.where(np_labels == i)
    #             loss = log_loss(
    #                 y_true=np_labels[idxs].tolist(), y_pred=preds[idxs], labels=labels
    #             )
    #             a = loss * len(np_labels[idxs].tolist()) / len(np_labels)
    #             print(
    #                 f"logloss for class {class_map[i]}: {loss}, contribution to total loss: {a}"
    #             )
    #             f.write(
    #                 f"logloss for class {class_map[i]}: {loss}, contribution to total loss: {a} \n"
    #             )
    #             losses.append(a)
    #         print(sum(losses))
    #         f.write(f"total loss: {str(sum(losses))}")

        # config["CFG"][f"logloss{ema}"] = logloss
        # config["CFG"][f"acc{ema}"] = float(acc)
        # with open(f"{root_dir}/config.json", "w") as f:
        #     json.dump(config, f)
    Test[vocab] = pd.DataFrame(preds)
    exp_list = ["ID"]
    exp_list.extend(vocab)
    Test[exp_list].to_csv(
        f"{root_dir}/submission{ema}_{fold}{logloss}.csv", index=False
    )


# %%
if CFG.weighted_dl:
    if CFG.soft_labels:
        counts = {
            col: Train_df.loc[
                (Train_df[col] > 0.69) & (Train_df["is_valid"] == False)
            ].shape[0]
            for col in vocab
        }
        counts["ND_soft"] *= 2
        counts["other_soft"] *= 2
        wgts = [
            1 / counts[label + "_soft"]
            for label in Train_df.loc[(Train_df["is_valid"] == False)].label_id.tolist()
        ]
    else:
        counts = Train_df[Train_df["is_valid"] == False]["label_id"].value_counts()
        counts["ND"] *= 2
        counts["other"] *= 2
        wgts = [
            1 / counts[label]
            for label in Train_df.loc[(Train_df["is_valid"] == False)].label_id.tolist()
        ]

dls = None


# %%
def create_learn(df, arc, item, batch, fold=None):
    dls = ImageDataLoaders.from_df(
        df,  # pass in train DataFrame
        valid_col="is_valid",
        seed=CFG.SEED,  # seed
        fn_col="path",  # filename/path is in the second column of the DataFrame
        label_col=(
            "label_id" if not CFG.soft_labels else vocab
        ),  # label is in the first column of the DataFrame
        label_delim=" " if CFG.loss_fn == "BCE" else None,
        y_block=(
            CategoryBlock(vocab=vocab)
            if not CFG.soft_labels
            else MultiCategoryBlock(vocab=list(vocab), encoded=True)
        ),
        bs=CFG.batch_size,  # pass in batch size
        num_workers=CFG.num_worker,
        item_tfms=item,  # pass in item_tfms
        batch_tfms=batch if not CFG.infer else Normalize.from_stats(*imagenet_stats),
        dl_type=WeightedDL if CFG.weighted_dl else TfmdDL,
        wgts=wgts if CFG.weighted_dl else None,
        pin_memory=True,
    )
    # metrics = (
    #     [accuracy, logloss_ml]
    #     if not CFG.soft_labels
    #     else (
    #         [accuracy_multi, logloss_ml] if not CFG.cv else [accuracy_multi, logloss_ml]
    #     )
    # )
    opt = get_optimizer(CFG.optimizer)
    if not isinstance(arc, str):
        learn = Learner(
            dls, arc, loss_func=get_loss_fn(CFG.loss_fn), opt_func=opt
        ).to_bf16()
    else:
        learn = vision_learner(
            dls,
            arc,
            loss_func=get_loss_fn(CFG.loss_fn),
            # metrics=metrics,
            opt_func=opt,
            n_out=5,
        ).to_bf16()
    if not CFG.infer:
        (
            learn.add_cb(WandbCallback(log_preds=False, seed=CFG.SEED))
            if CFG.wandb
            else None
        )
        (
            learn.add_cb(
                ReduceLROnPlateau(
                    monitor="log_loss" if not CFG.cv else "cross_entropy",
                    comp=np.less,
                    min_delta=CFG.reduce_lr_delta,
                    patience=1,
                    factor=2,
                    min_lr=1e-6,
                )
            )
            if CFG.reduce_lr_delta
            else None
        )
    learn.add_cb(EMACallback()) if CFG.EMA else None
    root_dir = f"models/{CFG.arc}/{id}"
    if AUG.mixup or AUG.cutmix:
        learn.loss_func.y_int = True
        learn.add_cb(CutMix(uniform=False)) if AUG.cutmix else None
        learn.add_cb(MixUp()) if AUG.mixup else None
    if CFG.continue_from:
        root_dir = f"models/{CFG.arc}/{CFG.continue_from}/{id}"
        CFG.freeze_epochs = 0
        f = CFG.continue_from.split("/")[-1]
        print(f"continuing from: {f}")
        model_name = "model_ema" if CFG.EMA else "model"
        if CFG.infer:
            root_dir = f"models/{CFG.arc}/{CFG.continue_from}"
        else:
            if os.path.exists(f"{root_dir}/{model_name}.pth") == False:
                shutil.copy2(
                    f"models/{arc}/{CFG.continue_from}/{model_name}.pth",
                    f"{root_dir}/{model_name}.pth",
                )
        learn.load(f"{arc}/{CFG.continue_from}/{model_name}")
        print(
            f"continuing {arc} from {CFG.continue_from}, starting from {CFG.start_epoch}th epoch"
        )
    print(learn)
    if CFG.compile:
        learn.add_cb(CompilerCallback())
    print(learn)
    if CFG.channels_last:
        learn.to_channelslast(amp_mode="bf16")
    # learn.add_cb(
    #     EarlyStoppingCallback2(
    #         monitor="cross_entropy" if not CFG.cv else "cross_entropy",
    #         patience=CFG.early_stop,
    #         comp=np.less,
    #     )
    # )

    if CFG.progressive_resize:
        learn.add_cb(ProgressiveResize(increase_by=32, empty_cache=True))
    learn.add_cb(
        CSVLogger(f"{root_dir}/history.csv", append=CFG.continue_from is not None)
    )
    root_dir = "/".join(root_dir.split("/")[1:])
    fold_str = f"_{fold}" if fold is not None else ""

    learn.add_cb(
        SaveModelCallback2(
            monitor="train_loss",
            comp=np.less,
            with_opt=True,
            fname=f"{root_dir}/model{fold_str}",
        )
    )
    learn.save = types.MethodType(save2, learn)
    # learn.add_cb(StopAtEpochCallback(max_epochs=CFG.max_epochs))
    learn.ep_start = CFG.start_epoch
    # with open(f"{root_dir}/config.json", 'w') as f:
    #         json.dump(config, f)
    return learn


# %%
xtra_tfms = []
augs = {
    "random_resized_crop_gpu": RandomResizedCropGPU(CFG.image_size, min_scale=0.6),
    "random_erasing_batch": RandomErasingBatch(p=0.4, sh=0.2, max_count=3),
    "normalize": Normalize.from_stats(*imagenet_stats),
}

for key in config["AUG"]:
    if key == "cutmix" or key == "mixup":
        continue
    if config["AUG"][key] == True and key != "aug_tfms":
        xtra_tfms.append(augs[key])
if config["AUG"]["aug_tfms"]:
    aug = aug_transforms(xtra_tfms=xtra_tfms, flip_vert=True, mult=2)
else:
    aug = xtra_tfms

arc = CFG.arc
if not CFG.cv:
    learner = create_learn(
        df=Train_df,
        arc=arc,
        item=Resize(CFG.image_size, method=CFG.resize_method),
        batch=aug,
    )

# %%
if CFG.infer:
    learner.to_fp32()
    if CFG.tta:
        tta_preds, targets = learner.tta(n=CFG.tta)
        preds = F.softmax(tta_preds, dim=1)
    else:
        preds, _ = learner.get_preds()
        preds = F.softmax(preds, dim=1)

    if CFG.EMA and not CFG.infer:
        log_infer(preds, ema="_ema_infer", write=True)
    else:
        log_infer(preds, ema="_infer", write=True)

# %%
# interp = Interpretation.from_learner(learner)
# interp.top_losses(9)

print(Train_df.iloc[0])
print(Train_df.iloc[-1])
print(CFG.arc)
print(print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad)))
# print(learner.summary())

print(f"Starting training {CFG.arc} for {CFG.epochs}")
print(f"using softlabel {CFG.soft_labels}") if CFG.soft_labels else None
print(f"using pseudolabel {CFG.pseudo_labels}") if CFG.pseudo_labels else None
# %%
import time

start = time.time()
if not CFG.infer and not CFG.cv:
    print(CFG.fit_method)
    if CFG.fit_method == "one_cycle":
        learner.fine_tune(CFG.epochs, CFG.lr, freeze_epochs=CFG.freeze_epochs)
    elif CFG.fit_method == "fit_flat_cos":
        # learner.freeze()
        # print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        # learner.fit(CFG.freeze_epochs, 2*CFG.lr)
        # learner.fit_one_cycle(CFG.freeze_epochs, slice(2*CFG.lr), pct_start=0.99)
        learner.unfreeze()
        print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        learner.fit_flat_cos(CFG.epochs, CFG.lr)
    elif CFG.fit_method == "fit_sgdr":
        # learner.freeze()
        # print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        # learner.fit_one_cycle(1, CFG.lr*2, pct_start=0.99)
        learner.unfreeze()
        print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        learner.fit_sgdr(n_cycles=3, cycle_len=CFG.epochs, lr_max=CFG.lr)
    elif CFG.fit_method == "fit":
        # learner.freeze()
        # print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        # learner.fit(3, , pct_start=0.99)
        learner.unfreeze()
        print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        # learner.fit(2, 5e-5)
        learner.fit(CFG.epochs, CFG.lr)
    elif CFG.fit_method == "warmup_one_cycle":
        learner.unfreeze()
        print(sum(p.numel() for p in learner.model.parameters() if p.requires_grad))
        # learner.fit(2, 5e-5)
        learner.fit(2, 5e-5)
        learner.fit_one_cycle(CFG.epochs, CFG.lr)

    if CFG.EMA and not CFG.infer:
        temp = learner.model
        learner.model = learner.best_model_ema
    learner.to_fp32()
    dl = learner.dls.test_dl(Test['path'])
    if CFG.tta:
        tta_preds, targets = learner.tta(dl = dl, n=CFG.tta)
        preds = F.softmax(tta_preds, dim=1)
    else:
        preds, _ = learner.get_preds(dl=dl)
        preds = F.softmax(preds, dim=1)

    if CFG.EMA and not CFG.infer:
        log_infer(preds, ema="_ema", write=True)
    else:
        log_infer(preds, ema="", write=True)
    learner.to_fp32()
    if CFG.EMA and not CFG.infer:
        learner.model = temp
        if CFG.tta:
            tta_preds, targets = learner.tta(n=CFG.tta)
            preds = F.softmax(tta_preds, dim=1)
        else:
            preds, _ = learner.get_preds()
            preds = F.softmax(preds, dim=1)
        log_infer(preds, ema="", write=True)

if not CFG.infer and CFG.cv:
    for fold in range(CFG.cv):
        Train_df["is_valid"] = False
        Train_df["is_valid"] = Train_df["fold"] == fold
        print(f"Training fold: {fold}")
        learner = None
        learner = create_learn(
            df=Train_df,
            arc=arc,
            item=Resize(CFG.image_size, method=CFG.resize_method),
            batch=aug,
            fold=fold,
        )
        learner.fine_tune(CFG.epochs, CFG.lr, freeze_epochs=CFG.freeze_epochs)

        if CFG.EMA and not CFG.infer:
            temp = learner.model
            learner.model = learner.best_model_ema

        if CFG.tta:
            dl = learner.dls.valid
            tta_preds, targets = learner.tta(dl, n=CFG.tta)
            preds = F.softmax(tta_preds, dim=1)
        else:
            preds, _ = learner.get_preds(dl)
            preds = F.softmax(preds, dim=1)

        if CFG.EMA and not CFG.infer:
            log_infer(preds, ema="_ema", fold=f"{fold}_", write=True)
        else:
            log_infer(preds, ema="", fold=f"{fold}_", write=True)

        if CFG.EMA and not CFG.infer:
            learner.model = temp
            if CFG.tta:
                tta_preds, targets = learner.tta(dl=dl, n=CFG.tta)
                preds = F.softmax(tta_preds, dim=1)
            else:
                preds, _ = learner.get_preds(dl=dl)
                preds = F.softmax(preds, dim=1)
            log_infer(preds, ema="", fold=f"{fold}_", write=True)


elapsed = time.time() - start
print("training and tta took: ", elapsed)
print(time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)))
# %%
if not CFG.infer:
    if CFG.continue_from:
        config["CFG"]["nth_epoch"] = learner.n_epoch - learner.cbs[5].wait
        with open(f"models/{CFG.arc}/{CFG.continue_from}/{id}/config.json", "w") as f:
            json.dump(config, f)
    else:
        with open(f"models/{CFG.arc}/{id}/config.json", "w") as f:
            json.dump(config, f)


# %%
if CFG.DEBUG == True:
    os.system(f"rm -rf {root_dir}")
# %%
# interp = Interpretation.from_learner(learner)
# interp.top_losses(9)

# %%
# interp.plot_top
