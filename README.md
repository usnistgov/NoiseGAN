## <u>**Evaluating Convolutional Generative Adversarial Networks with Classical Random Process Noise Models**</u>

## Overview
This repository contains Python code to execute experiments on deep generative
modeling of classical random process models for noise time series.  Specifically,
it includes Pytorch implementations of two generative adversarial network (GAN)
models for time series based on convolutational neural networks (CNNs):
**WaveGAN**, a 1-D CNN model, and **STFT-GAN**, a 2-D CNN model.  In addition,
there are methods for generating and evaluating noise time series defined several by classical random process models:
 - band-limited thermal noise, i.e., bandpass filtered white Gaussian noise
 - power law (fractional, colored) noise, including fractionally differenced white noise (FDWN), fractional Gaussian noise (FGN), and fractional Brownian motion (FBM)
 - generalized shot noise, including options for different pulse types and pulse amplitude distributions
 - impulsive noise, including Bernoulli-Gaussian (BG) and symmetric alpha stable (SAS) distributions

## Getting Started
The software enables automated testing of many model configurations across
different datasets.  Model creation and training is implemented using the Pytorch
library. This repository contains code for initializing experiment test runs (`main.py`),
training of GAN models(`gan_train.py`), loading target distributions (`data_loading.py`),
and evaluation(`gan_evaluation.py`) of generated distributions. The `/utils` directory
contains supporting modules for target dataset creation and model evaluation.
The `models/` directory contains modules that implement GAN architectures.

Synthetic training and test target datasets are created by running
`./utils/noise_dataset.py`, which executes `main()` in `noise_dataset.py`
as a script.  Datasets are saved to the directory `./Datasets/`.
The ranges of the loops in the script can be modified to create different
synthetic noise datasets.

Running `main.py` executes the GAN with model settings specified by the configuration
dictionary `training_specs_dict.py`.  Descriptions for
the fields specified in `training_specs_dict.py` are
located in the spreadsheet `./experiment_resources/trainspecs_dictionary_description.xlsx`.
Additionally, a set of model configurations can be run in an automated fashion by passing a
configuration table (csv file) as an argument to the main python module
(ex. `main.py --configs ./experiment_resources/test_configs.csv`).  Column labels of a
configuration table correspond to desired keys in the GAN configuration
dictionary that are to be changed across runs.

When running the models, experimental results are saved in `./model_results/`
with subdirectories named by their target dataset and other non-default
configurations and a time-stamp. Evaluation of the model is set to run at the
termination of model training.  Each test-run folder contains saved GAN models,
training metadata, as well as evaluations of the generated distributions.  
Aggregated plots across model runs are created using the scripts located in `./plotting_scripts/`.

Creation and evaluation of fractionally differenced white noise (FDWN)
datasets utilizes the `arfima` package in R.  We run R functions from python
using the interface provided by the python `rpy2` package.  Therefore, using any
methods for FDWN requires installing R and the `arfima` package.

## <u>Authors</u>
Jack Sklar (jack.sklar@nist.gov), Adam Wunderlich (adam.wunderlich@nist.gov)  \
Communications Technology Laboratory \
National Institute of Standards and Technology \
Boulder, Colorado

## <u>Licensing Statement</u>
This software was developed by employees of the National Institute of Standards and Technology (NIST), an
agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United
States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.
This software may be subject to foreign copyright.  Permission in the United States and in foreign countries,
to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this
software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this
notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY,
INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY
THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN
NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT
BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR
OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE
OR SERVICES PROVIDED HEREUNDER.
