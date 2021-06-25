# Accurate Classifiers are better Open Set Rcognition Models
This repository contains source code for the paper "Accurate Classifiers are better Open Set Rcognition Models"

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

#### CUDA Userspace Libraries
You might have to install a cudatoolkit with a version that matches your nvidia driver. 
You can find out the current nvidia-driver cuda api version with `nvidia-smi`. Afterwards, you can 
install the matching nvidia user space libraries with 
``` sh 
conda install -c pytorch cudatoolkit=<VERSION>
```

## Run Experiments
Theoretically, you should only have to run 
```shell
bin/experiment-vary-encoder.sh
```
