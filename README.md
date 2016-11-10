# Bloom Analysis

Idenifies and removes sub OTUs which grow under room temperature storage conditions.

## Installion

Download the repository from [GitHub](https://github.com/knightlab-analyses/bloom-analyses), and navigate into that directory. The repository can be installed by running the following commands in the linux terminal.

```
conda create -n bloom pip python=3 numpy jupyter seaborn matplotlib=1.5.1 statsmodels
source activate bloom
pip install scikit-bio==0.5.0
pip install git+https://github.com/amnona/heatsequer.git
pip install emperor --pre
pip install -e .
```
## Notebooks
All analysis notebooks can be initialized by running the command `jupyter notebook` from the terminal.

* [**ag\_alpha\_diversity.ipynb**](ipynb/ag\_alpha\_diversity.ipynb): Evaluates the effect of the identified bloom sequences on the difference in alpha diversity in the [American Gut Project](http://americangut.org/) data
* [**bloom_example.ipynb**](ipynb/bloom_example.ipynb) Demonstrates how the technique can be applied to filter out sequences in studies
* [**effect-of-blooms-on-bray-curtis-distance.ipynb**](ipynb/effect-of-blooms-on-bray-curtis-distance.ipynb): Evaluates changes in beta diversity between samples shipped through local post and samples that were frozen shortly after collection.
* [**identify-candidate-blooms.ipynb**](ipynb/identify-candidate-blooms.ipynb): Identifies the bloom sequences in the American Gut data based on comparison with storage studies and fresh frozen samples.