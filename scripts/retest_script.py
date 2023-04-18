#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:01:06 2023

@author: ajw2
"""

import os
import json
import sys
sys.path.insert(0,'./')
from gan_evaluation import retest_gan

FGN_parent_path = "./model_results/FGN/"
#FGN_dataset_paths = ["FGN_fixed_H5/", "FGN_fixed_H10/", "FGN_fixed_H20/", "FGN_fixed_H30/", "FGN_fixed_H40/", "FGN_fixed_H50/",
#                     "FGN_fixed_H60/", "FGN_fixed_H70/", "FGN_fixed_H80/", "FGN_fixed_H90/", "FGN_fixed_H95/"]
FGN_dataset_paths = ["FGN_fixed_H5/"]

FGN_stft128_paths = []
FGN_wave_paths = []
for FGN_path in FGN_dataset_paths:
    path = os.path.join(FGN_parent_path, FGN_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            FGN_wave_paths.append(model_path)
        else:
            if train_specs_dict['dataloader_specs']['dataset_specs']['nperseg'] == 128:
                FGN_stft128_paths.append(model_path)

for path in FGN_stft128_paths:
    print(f"retesting {path}")
    retest_gan(path)