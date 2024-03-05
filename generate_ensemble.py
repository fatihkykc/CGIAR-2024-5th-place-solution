import numpy as np
import pandas as pd
import torch
from fastai.vision.all import accuracy
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")


val = pd.read_csv("Test3.csv")


def log_infer(preds, ema=""):
    class_map = "class_map.txt"

    with open(class_map, "r") as f:
        class_map = f.read()
    class_map = class_map.split("\n")
    for i, cls in enumerate(class_map):
        val.loc[val["label_id"] == cls, "label_id"] = i

    labelz = val["label_id"].tolist()
    logloss = log_loss(y_true=labelz, y_pred=preds, labels=[0, 1, 2, 3, 4])
    print("logloss test:", logloss)
    acc = accuracy(preds, torch.tensor(labelz))
    print("acc test: ", acc)

    # calc logloss for each class
    np_labels = val["label_id"].to_numpy()
    losses = []
    for i in range(5):
        idxs = np.where(np_labels == i)
        loss = log_loss(
            y_true=np_labels[idxs].tolist(), y_pred=preds[idxs], labels=[0, 1, 2, 3, 4]
        )
        a = loss * len(np_labels[idxs].tolist()) / len(np_labels)
        print(
            f"logloss for class {class_map[i]}: {loss}, contribution to total loss: {a}"
        )
        losses.append(a)
    print(sum(losses))


preds = [
    #"models/vit_base_patch16_clip_384/submission_0.5429878069125621.csv",
    #"models/vit_base_patch16_clip_384/41fnrh4u/submission_0.5565204507545602.csv",
    #"models/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k/axv2u7jx/submission_0.5376033892313601.csv",
    "models/vit_base_patch16_clip_384/2nd/submission_0.5401759125577102.csv",
    "models/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k/3rd/submission_0.5407690905661905.csv",
]
weights = [0.5,0.5]
# weights = [0.2, 0.4, 0.4]
results = []
for i, pred in enumerate(preds):
    df = pd.read_csv(pred)
    if "DR" in df.columns:
        voc=["DR", "G", "ND", "WD", "other"]
        df.rename(
            columns={
                "WD": "DR",
                "G": "G",
                "DR": "ND",
                "ND": "WD",
                "other": "other",
            },
            inplace=True,
        )
    else:
        voc=["DR_soft", "G_soft", "ND_soft", "WD_soft", "other_soft"]
    results.append(
        torch.tensor(

            df[voc].to_numpy()
            * weights[i]
        )
    )


log_infer(torch.stack(results).sum(axis=0))

df.rename(
            columns={
                "DR_soft": "DR",
                "G_soft": "G",
                "ND_soft": "ND",
                "WD_soft": "WD",
                "other_soft": "other",
            },
            inplace=True,
        )
test = df[['ID','DR','G','ND','WD','other']]
test[['DR','G','ND','WD','other']] = torch.stack(results).sum(axis=0).cpu().numpy()
test.to_csv('ens.csv', index=False)
