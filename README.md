Interpreting linear Riemannian tangent space (RTS) models
=========================================================

This repository contains code for a EMBC 2021 submitted article
**On the interpretation of linear Riemannian tangent space model parameters in M/EEG**

The repository was forked on the repository:
https://github.com/DavidSabbagh/meeg_power_regression

Python dependencies
-------------------

 - numpy >= 1.20
 - scipy >= 1.12
 - matplotlib >= 3.3
 - scikit-learn >= 0.23
 - pandas >= 1.2.1
 - mne >= 0.22
 - h5py >= 3.1
 - pyriemann >= 0.2


Repository structure
-----------------

The root directory contains Python scripts that can be run to simulations (*sim_\*.py*) and analyze a publicly available MEG dataset (*dscmc_regression.py*) that was recorded to study cortico-muscular coherence (CMC).
The plots for the paper were generated using R. The scripts end with (*_plots.R*).
Some paths and plotting options are defined in the configuration files (*config.\[py|r\]*)

The *library* folder contains several utility functions and classes that are used in the analysis scripts.
