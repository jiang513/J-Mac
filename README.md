# J-Mac: Jacobian Matrix Meets Masked Contrastive Learning in Reinforcement Learning
This is a PyTorch implementation of **SVEA-J-Mac** and **DrQ-J-Mac**.
## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:
```
cd ./DMC-GB
conda env create -f ./setup/conda.yaml
conda activate dmcgb
sh ./setup/install_envs.sh
```
## Datasets
Part of this repository relies on external datasets. SVEA uses the Places dataset for data augmentation, which can be downloaded by running
```
[wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar](wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar)
```
## Training & Evaluation
In the `DMC-GB` and `VB-RMB` directories, `scripts` directories contain bash scripts for **SVEA-J-Mac** and **DrQ-J-Mac**, which can be run by `sh /DMC-GB/scripts/svea-J-Mac.sh` and `sh /VB-RMB/scripts/DrQ-J-Mac.sh` respectively.

Alternatively, you can call the python scripts directly, e.g. for training of **SVEA-J-Mac** call
```
python3 DMC-GB/src/train.py --seed 0 --algorithm svea --use_aux --use_jacobian
```
to run **SVEA-J-Mac** on the default task, `walker_walk`, and using the default hyperparameters.
