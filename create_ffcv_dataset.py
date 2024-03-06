import os

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-12.3/bin/ptxas"

from fastai.vision.all import *
from fastxtend.ffcv.all import *
from fastxtend.vision.all import *


class CFG:
    soft_labels = "vit_base_patch16_clip_384/ycozmy30"
    pseudo_labels = "vit_base_patch16_clip_384/ycozmy30"
    DEBUG = False
    relabeled = False


def splitter(df):
    train = df.index[df["is_valid"] == False].tolist()
    val = df.index[df["is_valid"] == True].tolist()
    return train, val


TRAIN = f"models/{CFG.soft_labels}/Train_soft_0.7.csv"
TEST = "Test.csv"
VAL = "Test3.csv" if not CFG.soft_labels else "test_soft.csv"
print(TRAIN)
print(VAL)
if CFG.DEBUG:
    Train = pd.read_csv(TRAIN, nrows=100)
    Test = pd.read_csv(TEST, nrows=100)
    val = pd.read_csv(VAL, nrows=100)
else:
    Train = pd.read_csv(TRAIN)
    Test = pd.read_csv(TEST)
    val = pd.read_csv(VAL)
Train = Train[Train["is_test"] == 0] if CFG.relabeled else Train
# Train.drop(["is_test","fname","leak_fname"], axis=1, inplace=True)


vocab = Train.label_id.unique()
print(vocab)
val["is_valid"] = True
Train["is_valid"] = False

Train_df = pd.concat([Train, val]).reset_index(drop=True)
Test_df = Train[Train["is_valid"] == True]

import numpy as np

vocab = ["DR_soft", "G_soft", "ND_soft", "WD_soft", "other_soft"]
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
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(vocab=list(vocab), encoded=True)),
    splitter=splitter,
    get_y=ColReader(vocab),
    get_x=ColReader("path"),
)
dset = dblock.datasets(Train_df)
path = Path.home() / ".cache/fastxtend"
path.mkdir(exist_ok=True)

rgb_dataset_to_ffcv(
    dset.valid,
    path / "cgiar_384_valid.ffcv",
    min_resolution=384,
    label_field=LabelField.float,
)
rgb_dataset_to_ffcv(
    dset.train,
    path / "cgiar_384_train.ffcv",
    min_resolution=384,
    label_field=LabelField.float,
)
