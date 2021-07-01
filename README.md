
<div align="center">    

# Accurate Classifiers are better Open Set Rcognition Models 

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.arxiv.org)
[![Conference](http://img.shields.io/badge/BMVC-2021-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
-->

</div>

## Description
This repository contains source code for the paper *Accurate Classifiers are better Open Set Rcognition Models* submitted to the BMVC.

Runs on Python 3.8, PyTorch and PyTorch lightning. Hyperparameter Optimization is done with Ray Tune.

## Setup
Create an anaconda environment:
``` sh 
conda env create --name osr
conda activate osr
```

You might need to add `src/` to your `PYTHONPATH`.
``` sh 
export PYTHONPATH="$PYTHONPATH:src/"
```

**CUDA Userspace Libraries**

You might have to install a cudatoolkit with a version that matches your nvidia driver. 
You can find out the current nvidia-driver cuda api version with `nvidia-smi`. Afterwards, you can 
install the matching nvidia user space libraries with 
``` sh 
conda install -c pytorch cudatoolkit=<VERSION>
```

## How to run   

You can train models with the command line interface:
```shell
src/train.py <config> [overrides]
```
The config contains configuration values and hyper parameters.

For example:
```shell
src/train.py --test config/cifar-10/resnet-18/softmax.yaml
```
Will run train and evaluate the resnet-18 model on the CIFAR-10 dataset.

Files will be located in `data/experiments/cifar-10/softmax/<date>`.


### Experiments
To replicate the experiments, run:
```shell
bin/experiment-vary-encoder.sh
```

### Pretrained Models

Pretrained models can be found in `data/pretrained`. 


<!--
## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={2021}
}
```
-->