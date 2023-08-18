## <u>**NoiseGAN: Software for Evaluating Convolutional Generative Adversarial Networks with Classical Random Process Noise Models**</u>

## Overview
This repository contains Python code to execute experiments on deep generative
modeling of noise time series.  Specifically, it includes Pytorch implementations of two generative adversarial network (GAN) models for time series based on convolutational neural networks (CNNs): **WaveGAN**, a 1-D CNN model, and **STFT-GAN**, a 2-D CNN model.  In addition, there are methods for generating and evaluating noise time series defined several by classical random process models:

 - band-limited thermal noise, i.e., bandpass filtered white Gaussian noise
 - power law (fractional, colored) noise, including fractional Gaussian noise (FGN), fractional Brownian motion (FBM), and fractionally differenced white noise (FDWN)
 - generalized shot noise, including options for different pulse types and pulse amplitude distributions
 - impulsive noise, including Bernoulli-Gaussian (BG) and symmetric alpha stable (SAS) distributions

## Reference
A. Wunderlich, J. Sklar, "Data-Driven Modeling of Noise Time Series with Convolutional Generative Adversarial Networks", Machine Learning: Science and Technology, 2023, https://doi.org/10.1088/2632-2153/acee44

## Getting Started
The software enables automated testing of many model configurations across
different datasets.  Model creation and training is implemented using the Pytorch
library. This repository contains code for initializing experiment test runs (`main.py`),
training of GAN models(`gan_train.py`), loading target distributions (`data_loading.py`),
and evaluation(`gan_evaluation.py`) of generated distributions. The `utils/` subdirectory contains supporting modules for target dataset creation and model evaluation.  The `models/` subdirectory contains modules that implement GAN architectures.

Synthetic training and test target datasets are created by running
`utils/noise_dataset.py`, which executes `main()` in `noise_dataset.py`
as a script.  Datasets are saved to the subdirectory `Datasets/`. The ranges of the loops in the script can be modified to create different synthetic noise datasets.

Running `main.py` executes the GAN with model settings specified by the configuration dictionary `training_specs_dict.py`.  Descriptions for the fields specified in `training_specs_dict.py` are
located in the spreadsheet `experiment_resources/trainspecs_dictionary_description.xlsx`.
Additionally, a set of model configurations can be run in an automated fashion by passing a
configuration table (csv file) as an argument to the main python module, e.g., `main.py --configs ./experiment_resources/test_configs.csv`.  Column labels of a
configuration table correspond to desired keys in the GAN configuration
dictionary that are to be changed across runs.

When running the models, experimental results are saved in `model_results/`
with subdirectories named by their target dataset and other non-default
configurations and a time-stamp. Evaluation of the model is set to run at the
termination of model training.  Each test-run folder contains saved GAN models,
training metadata, as well as evaluations of the generated distributions.
Aggregated evaluation plots across model runs are created using the scripts located in `scripts/`.

Packages in our python environment are documented by `conda_requirements.txt` and `pip_requirements.txt` in the `experiment_resources/` subdirectory, which were created with the `conda list` and `pip freeze` commands, respectively.  

To use methods for generating and evaluating fractionally differenced white noise (FDWN), it is necessary to uncomment the associated block of package imports at the top of `utils/fractional_noise_utils.py`.  The FDWN methods utilize the `arfima` package in R, which can be run from python using the interface provided by the python `rpy2` package.  Therefore, using any methods for FDWN requires installing R and the `arfima` R package.
