import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-12.3/bin/ptxas"
import pandas as pd
from fastai.vision.all import *
from fastai import *
from fastxtend.vision.all import *

TRAIN = pd.read_csv('Train2.csv')
TEST = pd.read_csv('Test2.csv')

arc ='vit_base_patch16_clip_384'
continue_from=""
EMA=False
SEED=0
resize_method='squish'
image_size = 384
loss_fn ='CE'
optimizer='AdamW'


# from lion_pytorch import Lion
def get_optimizer(opt):
    optimizer=None
    if opt == "lion":
        # torch_lion = partial(OptimWrapper, opt=Lion)
        # optimizer = partial(torch_lion)
        optimizer = lion(foreach=True)
    if opt=="Adam":
        t_adam = partial(OptimWrapper, opt=torch.optim.Adam)
        optimizer = partial(t_adam, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    elif opt=="AdamW":
        # adamw = partial(OptimWrapper, opt=torch.optim.AdamW)
        # optimizer = partial(adamw, betas=(0.9, 0.999), eps=1e-08)
        optimizer = adam(foreach=True)
    elif opt=="SGD":
        optimizer = partial(torch.optim.SGD)
    elif opt=="RMSprop":
        optimizer = partial(torch.optim.RMSprop, eps=1e-08)
    return optimizer

def get_loss_fn(loss_name):
    if loss_name == 'Focal':
        loss=FocalLoss()
    if loss_name =='CE':
        loss=nn.CrossEntropyLoss(label_smoothing=0)
    if loss_name=='LBCE':
        loss=LabelSmoothingCrossEntropy()
    if loss_name=='BCE':
        loss=BCEWithLogitsLossFlat()
    return loss

set_seed(SEED)

def load_model():
    model_name='model'
    opt = get_optimizer(optimizer)
    damage = TRAIN.label_id.unique()
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=damage)),
                get_x = ColReader('path'),
                get_y = ColReader('label_id'),
                item_tfms=Resize(image_size, method=resize_method),
                batch_tfms=Normalize.from_stats(*imagenet_stats)
                )
    dls = dblock.dataloaders(TRAIN, bs=64, seed=SEED, drop_last=False, shuffle=False)
    dls.rng.seed(SEED)
    learn = vision_learner(dls, arc, loss_func=get_loss_fn(loss_fn), opt_func=opt).to_bf16()
    if EMA:
        model_name='model_ema'
    learn.load(f'{arc}/{continue_from}/{model_name}')
    return learn

learner= load_model()

test_dl = learner.dls.test_dl(TRAIN['path'])
preds, _ = learner.get_preds(dl=test_dl)
preds=F.softmax(preds, dim=1)

from sklearn.metrics import log_loss
class_map = 'class_map.txt'
with open(class_map, 'r') as f:
    class_map = f.read()
class_map = class_map.split('\n')
TRAIN['label_encode'] = 0
for i,cls in enumerate(class_map):
    TRAIN.loc[TRAIN['label_id']==cls, 'label_encode']=i
    
    
labelz = TRAIN['label_encode'].tolist()
logloss = log_loss(y_true=labelz, y_pred=preds, labels=[0,1,2,3,4])
print("logloss train:", logloss)

trn = pd.concat([TRAIN, pd.get_dummies(TRAIN['label_id'], dtype='int') ], axis=1)
for i, cls in enumerate(class_map):
    pred_labels = preds[:, i].numpy() * 0.3
    hard_labels = trn[cls] * 0.7
    TRAIN[cls+'_soft'] = pred_labels+hard_labels

TRAIN.to_csv(f'models/{arc}/{continue_from}/Train_soft_0.7.csv', index=False)

print(f"train softlabels saved to models/{arc}/{continue_from}/Train_soft_0.7.csv")
