### General pipeline is as the following:

- Train a vit-base model on the whole dataset
- Generate softlabels and pseudolabels using the Trained model
- Train another vit model with the pseudolabels and the softlabels
- Train an eva02_Base model with the pseudolabels and the softlabels
- Ensemble the last 2 vit and eva02 model and submit

### What I did and did not figured out about the competition and the dataset

- use pseudolabels and softlabels
- use ffcv for gpu utilization, this allows training larger models.
- TTA works good.
- Could not figure out which augmentations work best, needed to experiment more
- Did not used cross validation for my submissions, since the time it takes is a lot, I wanted to give ensembling the edge and trained my models on the whole dataset when submitting.
- Onecycle lr worked the best, both in torch and fastai implementations.
- Transformer models were a lot better than convolutionals for my configuration

