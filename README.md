# MoCo-CIFAR10
### Preparation
Download [CIFAR-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and unzip it into ```./data/```
### Pretrain
```bash
python demo.py --results-dir [your results directory] --tensorboard
```
### Train
```bash
python demo_lincls.py --resume [your pretrained model] --results-dir [your results directory] --tensorboard
```
