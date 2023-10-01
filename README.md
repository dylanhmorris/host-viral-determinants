# Host and viral determinants of airborne transmission of SARS-CoV-2 in the Syrian hamster

Julia R. Port, Dylan H. Morris, Jade C. Riopelle, Claude Kwe Yinda, Victoria A. Avanzato, Myndi G. Holbrook, Trenton Bushmaker, Jonathan E. Schulz, Taylor A. Saturday, Kent Barbian, Colin A. Russell, Rose Perry-Gottschalk, Carl I. Shaia, Craig Martens, James O. Lloyd-Smith, Robert J. Fischer, Vincent J. Munster

## Repository information
This repository accompanies the article "Host and viral determinants of airborne transmission of SARS-CoV-2 in the Syrian hamster" (JR Port et al.). It provides code for reproducing Bayesian inference analyses from the paper and producing display figures.


## License and citation information
If you use the code or data provided here, please make sure to do so in light of the project [license](LICENSE) and please cite our work as below:

- J.R. Port et al. Host and viral determinants of airborne transmission of SARS-CoV-2 in the Syrian hamster. 2023. https://doi.org/10.7554/eLife.87094.

Bibtex record:
```
@article{port2023hostviral,
  title={Host and viral determinants of airborne transmission of SARS-CoV-2 in the Syrian hamster},
  author={Port, Julia R and Morris, Dylan H and Yinda, Claude Kwe and Riopelle, Jade C and Avanzato, Victoria A and Holbrook, Myndi G and Schukz, Jonathan E and Saturday, Taylor A and Bushmaker, Trenton and Barbian, Kent and Russell, Colin A. and Perry-Gottschalk, Rose and Shaia, Carl I and Martens, Craig and Lloyd-Smith, James O and Fischer, Robert J and Munster Vincent J},
  journal={{eLife} reviewed preprint},
  year={2023},
  doi={https://doi.org/10.7554/eLife.87094}
}
```

## Article abstract 
It remains poorly understood how SARS-CoV-2 infection influences the physiological host factors important for aerosol transmission. We assessed breathing pattern, exhaled droplets, and infectious virus after infection with Alpha and Delta variants of concern (VOC) in the Syrian hamster. Both VOCs displayed a confined window of detectable airborne virus (24h - 48h), shorter than compared to oropharyngeal swabs. The loss of airborne shedding was linked to airway constriction resulting in a decrease of fine aerosols produced. Male sex was associated with increased viral replication and virus shedding in the air, including a VOC-independent particle-profile shift towards smaller droplets. Transmission efficiency varied among donors, including a superspreading event. Co-infection with VOCs only occurred when both viruses were shed by the same donor during an increased exposure timeframe. This highlights that assessment of host and virus factors resulting in a differential exhaled particle profile is critical for understanding airborne transmission.


## Directories
- ``src``: all code, including numerics and figure generation:
- ``out``: mcmc output files
- ``ms/figures`: figures from the manuscript

# Reproducing analysis

A guide to reproducing the analysis from the paper follows. Code for the project should work on most standard macOS, Linux, and other Unix-like systems. It has not been tested on Windows. If you are using a Windows machine, you can try running the project within a [WSL2](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) environment containing Python 3.

You will also need a working $\TeX$ installation to render the text for the figures as they appear in the paper. If you do not have $\TeX$, you can either:
1. Install [TeXLive](https://tug.org/texlive/) (or another $\TeX$ distribution)
2. Turn off $\TeX$-based rendering by setting ``mpl.rcParams['text.usetex'] = False`` in this project's `src/plotting.py` file.

## Getting the code
First download the code. The recommended way is to ``git clone`` our Github repository from the command line:

    git clone https://github.com/dylanhmorris/host-viral-determinants.git

Downloading it manually via Github's download button should also work.

## Basic software requirements

The analysis can be auto-run from the project `Makefile`, but you may need to install some external dependencies first. In the first instance, you'll need a working installation of Python 3 (tested on Python 3.10 and 3.11) with the package manager `pip` and a working installation of Gnu Make or similar. Verify that you do by typing `which make` and `which pip` at the command line. 

## Virtual environments
If you would like to isolate this project's required dependencies from the rest of your system Python 3 installation, you can use a Python [virtual environment](https://docs.python.org/3/library/venv.html). 

With an up-to-date Python installation, you can create one by running the following command in the top-level project directory.

```
python3 -m venv .
```

Then activate it by running the following command, also from the top-level project directory.
```
source bin/activate
```

Note that if you close and reopen your Terminal window, you may need to reactivate that virtual environment by again running `source bin/activate`.

## Python packages
A few external python packages need to be installed. You can do so by typing the following from the top-level project directory.

    make install
    
or 

    pip install -r requirements.txt

## Running the analyses

The simplest approach is simply to type ``make`` at the command line, which should produce a full set of figures and results.

If you want to do things piecewise, typing ``make <path/to/filename>`` for any of the files present in the complete repository uploaded here should also work.

By default, the pipeline runs 3  CMC chains for each fit, with 1000 warmup steps and 1000 sample draws per chain. It runs them in serial by default when run on a CPU, but may run them in parallel if GPU or TPU is available. You can reconfigure things in `src/config.py`, but note that this may affect reproducibility.

Some shortcuts are available:

- ``make test`` runs unit tests via `pytest`
- ``make checks`` produces prior predictive checks
- ``make chains`` produces all MCMC output, including prior predictive checks
- ``make figures`` produces all figures
- ``make clean`` removes all generated files, leaving only source code (though it does not uninstall packages)
