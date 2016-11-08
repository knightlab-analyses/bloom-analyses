# bloom_analyses

A repository containing analyses used to perform bloom filtering

# Install

To reproduce the analyses in the analyses in `ipynb`, you will need to build the following enviroment. You will only need to do this once.

```
conda create -n bloom pip python=3 numpy jupyter seaborn matplotlib=1.5.1 h5py
source activate bloom
pip install biom-format
pip install scikit-bio
pip install -e .
```