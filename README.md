[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10232698.svg)](https://doi.org/10.5281/zenodo.10232698)

## *Converting an allocentric goal into an egocentric steering signal*

This repository contains code for producing the analyses and figures in: [Mussells Pires et al, *Converting an
allocentric goal into an egocentric steering signal*](https://www.biorxiv.org/content/10.1101/2022.11.10.516026v1)

The pipeline for analazing imaging, electrophysiology and behavioural data uses the raw data (axon binary files for
behavioural and electrophysiology data and registered two-photon tiff images) as a starting point. These data have not
been made available online due to the relatively large amount of space that would be required to store them, but can be
made available upon request. Minimally processed data (*F* timeseries for imaging ROIs, etc.) are available
here: https://doi.org/10.5281/zenodo.10145317.

### Repository structure

`notebooks/` contains a jupyter notebook that generates all the figures for each dataset:

- EPG_FC2.ipynb (Fig. 1)
- FC2_stimualtion.ipynb (Fig. 2)
- PFL3_ephys.ipynb (Fig. 3)
- Model.ipynb (Fig. 4)
- PFL3_LAL_imaging.ipynb (Fig. 5a-d)
- PFL3_LAL_stimualtion.ipynb (Fig. 5e-h)
- Wind_task.ipynb (Fig. 6)
- EM.ipynb (Extended Fig. 5)

`analysis/` <br>python code code for all behaviour, imaging, electrophysiology experiments & connectomic analyses (loading
data, analysis & plotting functions)

`MATLAB/` <br> MATLAB code for generating fits, the model & simulating silencing PFL3 neurons.

### Requirements

The majority of the analyses were perfomed using python, but some analyses were performed in MATLAB. All figures were produced using python. In some cases the output of the MATLAB analyses have been hardcoded into the relevant `.py` files. All jupyter notebooks, except EM.ipynb, should use the `python==3.6 analysis` environment (see installation instructions below). For connectomic analyses in EM.ipynb use
a `python==3.12` environment and install [neuprint](https://connectome-neuprint.github.io/neuprint-python/docs/).

### Installation instructions for `analysis` environment

The following instructions have been tested on an M1 chip mac

```
# create empty environment
conda create -n analysis

# activate
conda activate analysis

# use x86_64 architecture channel(s)
conda config --env --set subdir osx-64

# install python
conda install python=3.6

# orginal yaml was without builds 
conda env update --name analysis --file YOUR_PATH/analysis/analysis_environment.yml --prune

python -m ipykernel install --user --name=analysis
```
