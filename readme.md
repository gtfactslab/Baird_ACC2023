# Runtime Assurance with stlpy

This repository accompanies the paper submitted to ACC "Guaranteeing Signal Temporal Logic Safety Specifications with
Runtime Assurance" and may be found on [arxiv](https://arxiv.org/). The code necessary to generate the figures in our paper is contained in this repository.

See [stlpy](https://stlpy.readthedocs.io/en/latest/)'s documentation. In theory, our solution may be re-written to use MIT's Drake solver or scipy. We use gurobi.

## Pre-requsites
Python 3.6 or greater and must be installed.
The following packages are required (install with `pip3 install numpy` or your favorite package manager):
- numpy
- matplotlib
- scipy
- pytorch
- gurobi (see below)

## Installation
- Follow the instructions to install [gurobi](https://www.gurobi.com/downloads/free-academic-license/). It is free for academia.
- Clone this repo
- Clone [the stlpy repo](https://github.com/vincekurtz/stlpy)
- Replace the file `stlpy/solvers/gurobi/gurobi_micp.py` with the one in this repo
- Run `python setup.py install` from the home directory of the `stlpy` repo.

## Optional installations
- [MikTex](https://miktex.org/) or another LaTeX interpreter, for LaTeX to appear in PyPlot plots.

## Generating plots

Open a terminal, navigate to the repo, and run `python rta-stlpy.py`. By default, this will run the case dT=1, u=1 for all time.

For the other cases, open `rta-stlpy.py` and uncomment lines on the initial condition `x0`, input `u_nominal`, and discretization step `dT` as appropriate.