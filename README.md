# SIVCS
[zh](README_zh_CN.md) | [en](README.md)

This repo contains the implementation code for the paper "Size Invariant Visual Cryptography Schemes with Evolving Threshold Access Structures."
## Requirements
- python 3.9
- Jupyter Notebook is required to run testing scripts (*.ipynb). 

Install all libraries by running the command:
```shell
pip install jupyter && pip install -r requirements.txt
```
## Guide
### optimize.py
This file implements the core algorithms, with each algorithm corresponding to one of the following four functions:
- `optimize_nonliner`  Implements the optimization using the `scipy.optimize` for nonlinear programming. It can only calculate the solution for k=2.
- `optimize_sa1` Implements the basic simulated annealing algorithm with a single loop for iteration. This function is deprecated.
- `optimize_sa2` Implements the simulated annealing algorithm with an additional loop for the Markov chain.
- `optimize_sa3` Implements the simulated annealing algorithm with updates to hyperparameters after each iteration of the Markov chain.

The results are stored in the `OptimizedResult` object.

### opt_sa.ipynb
This script allows you to visualize the generated images.
### *_stat.ipynb
These scripts perform statistical analysis on results.