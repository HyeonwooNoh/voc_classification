# Setting python environment with anaconda

Following instructions are a manual for setting the python environment
with anaconda.
The manuals are about seting pytorch up, but currently we pytorch setting is
not a mendatory requirement.

1.  download anaconda3 4.2 (not 4.3) and run the bash script
```bash
$ wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
$ bash <downloaded file>
```

1. make virtual environment with anaconda
```bash
## Usage of conda
# create <name of env> with python <version>
$ conda create -n <name of env> python=<version>

# activate new environment(<name of env>)
$ conda activate <name of env>
# if you want to see all the list of virtual env you made, type 
$ conda info -e or $ conda info --envs

# for install packages in certain project
$ conda install -n <name of env> <package_name1>=<version> <package_name2>=<version>

# install a new package on current environment.(use this after source activate)
$ conda install <package_name>=<version>

# deactivate virtualenv
$ deactivate <name of env>
```

```bash
# for pytorch type bellow:
$ conda create -n pytorch-py3 python=3.5 numpy scipy matplotlib sphinx nose pillow jupyter nltk tqdm h5py pyyaml
$ source activate pytorch-py3

# Install packages from other channels
conda install -c conda-forge xmltodict=0.10.2
```

1. Install pytorch
```bash
# make sure you are in pytorch-py3
# visit pytorch.org
# python 3.5, CUDA 8.0
conda install pytorch torchvision cuda80 -c soumith
```
