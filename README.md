# LyaNet: A Lyapunov Framework for Training Neural ODEs

This repository depends on a few submodules. To clone either use ```git clone --recurse-submodules``` or use ```git submodule update --init --recursive``` after cloning.

Provide the model type```--config-name``` to train and test models 
configured as those shown in the paper. 

## Classification Training
For the code assumes the project root is the current directory.

Example commands:

```bash
python sl_pipeline.py --config-name classical +dataset=MNIST
```

Tensorboards are saved to ```run_data/tensorboards``` and can be viewed by 
running:

```bash
tensorboard --logdir ./run_data/tensorboards --reload_multifile True
```

Only the model with the best validation error is saved. To quickly verify 
the the test error of this model, run the adversarial robustness script. It 
prints the nominal test error before performing the attack.

## Adversarial Robustness

Assuming the current directory is ``robustness``. Notice that the model file 
name will be different depending on the dataset and model combination you 
have run. The path provided should provide an idea of the directory structure 
where models are stored. 

These scripts will print the testing error, followed by the testing error 
with and adversarial attack. Notice adversarial testing requires 
significantly more resources.

### L2 Adversarial robustness experiments

```bash
PYTHONPATH=../ python untargeted_robustness.py --config-name classical norm="2" \
+dataset=MNIST \
"+model_file='../run_data/tensorboards/d.MNIST_m.ClassicalModule(RESNET18)_b.128_lr.0.01_wd.0.0001_mepoch120._sd0/default/version_0/checkpoints/epoch=7-step=3375.ckpt'"
```

### L Infinity Adversarial robustness experiments

```bash
PYTHONPATH=../ python untargeted_robustness.py --config-name classical \
norm="inf"  +dataset=MNIST \
"+model_file='../run_data/tensorboards/d.MNIST_m.ClassicalModule(RESNET18)_b.128_lr.0.01_wd.0.0001_mepoch120._sd0/default/version_0/checkpoints/epoch=7-step=3375.ckpt'"
```

### Datasets supported

* ```MNIST```
* ```FashionMNIST```
* ```CIFAR10```
* ```CIFAR100```

### Models Supported

* ```anode``` : Data-controlled dynamics with ResNet18 Component trained 
  through solution differentiation
* ```classical```: ResNet18
* ```lyapunov```: Data-controlled dynamics with ResNet18 Component trained 
  with LyaNet
* ```continuous_net```: ContinuousNet from [1] trained through solution 
  differentiation
* ```continuous_net_lyapunov```: ContinuousNet from [1] trained with LyaNet 

## References
1. [Continuous-in-Depth Neural Networks](https://arxiv.org/abs/2008.02389) 
   [Code](https://github.com/afqueiruga/ContinuousNet)
2. [Learning by Turning: Neural Architecture Aware Optimisation](https://arxiv.org/abs/2102.07227)
[Code](https://github.com/jxbz/nero)
