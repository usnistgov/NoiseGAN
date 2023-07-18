#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:01:06 2023

@author: ajw2
"""

import os
import sys
sys.path.insert(0,'./')
from gan_evaluation import retest_gan

parent_paths = ["/data/noise-gan/model_results/bandpass/", "/data/noise-gan/model_results/BG/",
                "/data/noise-gan/model_results/FBM/", "/data/noise-gan/model_results/FGN/",
                "/data/noise-gan/model_results/SAS/", "/data/noise-gan/model_results/shot/"]

BPWN_dataset_paths = ["band0/", "band1/", "band2/", "band3/", "band4/", "band5/", "band6/", "band7/"]
BG_dataset_paths = ["BG_fixed_IP1/", "BG_fixed_IP5/", "BG_fixed_IP10/", "BG_fixed_IP15/",
                     "BG_fixed_IP20/", "BG_fixed_IP30/", "BG_fixed_IP40/", "BG_fixed_IP50/",
                     "BG_fixed_IP60/", "BG_fixed_IP70/", "BG_fixed_IP80/", "BG_fixed_IP90/"]
FGN_dataset_paths = ["FGN_fixed_H5/", "FGN_fixed_H10/", "FGN_fixed_H20/", "FGN_fixed_H30/", "FGN_fixed_H40/", "FGN_fixed_H50/",
                     "FGN_fixed_H60/", "FGN_fixed_H70/", "FGN_fixed_H80/", "FGN_fixed_H90/", "FGN_fixed_H95/"]
FBM_dataset_paths = ["FBM_fixed_H5/", "FBM_fixed_H10/", "FBM_fixed_H20/", "FBM_fixed_H30/", "FBM_fixed_H40/",
                     "FBM_fixed_H50/", "FBM_fixed_H60/", "FBM_fixed_H70/", "FBM_fixed_H80/", "FBM_fixed_H90/", "FBM_fixed_H95/"]
SAS_dataset_paths = ["SAS_fixed_alpha50/", "SAS_fixed_alpha60/", "SAS_fixed_alpha70/", "SAS_fixed_alpha80/",
                     "SAS_fixed_alpha90/", "SAS_fixed_alpha100/", "SAS_fixed_alpha110/", "SAS_fixed_alpha120/",
                     "SAS_fixed_alpha130/", "SAS_fixed_alpha140/", "SAS_fixed_alpha150/"]
SNGE_dataset_paths = ["shot_gaussian_exponential_fixed_ER25/", "shot_gaussian_exponential_fixed_ER50/",
                      "shot_gaussian_exponential_fixed_ER75/", "shot_gaussian_exponential_fixed_ER100/",
                      "shot_gaussian_exponential_fixed_ER125/", "shot_gaussian_exponential_fixed_ER150/",
                      "shot_gaussian_exponential_fixed_ER175/", "shot_gaussian_exponential_fixed_ER200/",
                      "shot_gaussian_exponential_fixed_ER225/", "shot_gaussian_exponential_fixed_ER250/",
                      "shot_gaussian_exponential_fixed_ER275/", "shot_gaussian_exponential_fixed_ER300/"]
SNOE_dataset_paths = ["shot_one_sided_exponential_exponential_fixed_ER25/", "shot_one_sided_exponential_exponential_fixed_ER50/",
                      "shot_one_sided_exponential_exponential_fixed_ER75/", "shot_one_sided_exponential_exponential_fixed_ER100/",
                      "shot_one_sided_exponential_exponential_fixed_ER125/", "shot_one_sided_exponential_exponential_fixed_ER150/",
                      "shot_one_sided_exponential_exponential_fixed_ER175/", "shot_one_sided_exponential_exponential_fixed_ER200/",
                      "shot_one_sided_exponential_exponential_fixed_ER225/", "shot_one_sided_exponential_exponential_fixed_ER250/",
                      "shot_one_sided_exponential_exponential_fixed_ER275/", "shot_one_sided_exponential_exponential_fixed_ER300/"]

run_dir_list = []
for i, parent_path in enumerate(parent_paths):
    match i:
        case 0:
            dataset_paths = BPWN_dataset_paths
        case 1:
            dataset_paths = BG_dataset_paths
        case 2:
            dataset_paths = FBM_dataset_paths
        case 3:
            dataset_paths = FGN_dataset_paths
        case 4:
            dataset_paths = SAS_dataset_paths
        case 5:
            dataset_paths = SNGE_dataset_paths + SNOE_dataset_paths
    for data_path in dataset_paths:
        path = os.path.join(parent_path, data_path)
        rundirs = next(os.walk(path))[1]
        for rundir in rundirs:
            model_path = os.path.join(path, rundir)
            run_dir_list.append(model_path)

for model_path in run_dir_list:
    print(f"retesting {model_path}")
    target_dir = '/data/noise-gan/Datasets/'
    retest_gan(model_path, target_dir)
