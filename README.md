# SnapshotEnsemble
Pytorch implementation of Snapshot Ensemble. 

Original Paper: [Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)

Snapshot ensemble is a method which constructs neural network ensemble from single model training. The core of this method is "forcing model to converge local minima for M times by propery tuned learning rate". This repository supports multiple dataset and ConvNets such as MNIST, CIFAR-10, CIFAR-100, ResNet, EfficientNet, ConvNet.

For training, please run this command on your machine.
```
bash train.sh
```
For evaluation, after specify the directory which contains model checkpoints(it will be wandb log dir) in eval.sh, run this command.
```
bash eval.sh
```