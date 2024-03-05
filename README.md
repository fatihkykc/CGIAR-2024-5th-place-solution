##

Total process is:

- Train a vit model on the whole dataset
- Generate softlabels and pseudolabels from the Trained model
- Train another vit model with the pseudolabels and the softlabels
- Train an eva02_Base model with the pseudolabels and the softlabels

ensemble the last 2 vit and eva02 model and submit