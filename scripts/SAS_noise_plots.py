#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:55:52 2023

@author: ajw2
"""

import os
import re
import json
import h5py
import sys
sys.path.insert(0,'../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.alpha_stable_noise_utils as asn

plot_path = "../paper_plots/"
if not os.path.exists(plot_path):
   os.makedirs(plot_path)

plot_separate_qt_results = False # separately plot results with quantile transformation
SAS_parent_path = "/data/noise-gan/model_results/SAS/"
SAS_dataset_paths = ["SAS_fixed_alpha50/", "SAS_fixed_alpha60/", "SAS_fixed_alpha70/", "SAS_fixed_alpha80/",
                     "SAS_fixed_alpha90/", "SAS_fixed_alpha100/", "SAS_fixed_alpha110/", "SAS_fixed_alpha120/",
                     "SAS_fixed_alpha130/", "SAS_fixed_alpha140/", "SAS_fixed_alpha150/"]

c = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
     "#920000","#924900","#db6d00","#24ff24","#ffff6d"]
c1, c2, c3, c4 =  c[2], c[6], c[10], c[12]

#%%

SAS_stft_fs_paths = [] # feature_min_max scaling
SAS_wave_fs_paths = [] # feature_min_max scaling
SAS_stft_gs_paths = [] # global_min_max scaling
SAS_wave_gs_paths = [] # global_min_max scaling
SAS_stft_qt_paths = [] # quantile transform scaling
SAS_wave_qt_paths = [] # quantile transform scaling
for SAS_path in SAS_dataset_paths:
    path = os.path.join(SAS_parent_path, SAS_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            if train_specs_dict['dataloader_specs']['dataset_specs']['quantize'] == 'channel':
                SAS_wave_qt_paths.append(model_path)
            else: # no quantile tranformation
                if train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'feature_min_max':
                    SAS_wave_fs_paths.append(model_path)
                elif train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'global_min_max':
                    SAS_wave_gs_paths.append(model_path)
        else: # stftgan
            if train_specs_dict['dataloader_specs']['dataset_specs']['quantize'] == 'channel':
                SAS_stft_qt_paths.append(model_path)
            else: # no quantile tranformation
                if train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'feature_min_max':
                    SAS_stft_fs_paths.append(model_path)
                elif train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'global_min_max':
                    SAS_stft_gs_paths.append(model_path)

#%%

test_set_groups = [SAS_wave_fs_paths, SAS_stft_fs_paths, SAS_wave_qt_paths, SAS_stft_qt_paths, SAS_wave_gs_paths, SAS_stft_gs_paths]
model_types = ["wavegan", "stftgan", "wavegan_qt", "stftgan_qt", "wavegan_gs", "stftgan_gs"]
noise_set_names = ["SAS", "SAS", "SAS", "SAS", "SAS", "SAS"]

df_temp = []
for test_set, noise_set, model in zip(test_set_groups, noise_set_names, model_types):
    for model_path in test_set:
        with open(os.path.join(model_path, "summary_metrics.json")) as f:
            data = json.load(f)
            data["config"] = model_path
            data["model_type"] = model
            data["noise_type"] = noise_set
            with open(os.path.join(model_path, "gan_train_config.json")) as f2:
                train_dict = json.load(f2)
                data['dataset'] = train_dict['dataloader_specs']['dataset_specs']['data_set']
            #print(f"{data['dataset']}, {model}, {noise_set}")
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(os.path.join(plot_path, "SAS_results.csv"), index=False)

print(metrics_df["model_type"].value_counts())


#%%

if plot_separate_qt_results:
    # plot quantile transform results separately

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize = (5,7))
    noise_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    model_metrics_df = metrics_df[metrics_df["noise_type"] == "SAS"]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan_qt"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan_qt"]
    box_range = list(range(len(stft_metrics_df)))
    target_range = [pos - 0.2 for pos in box_range]
    stft_range = [pos + 0.2 for pos in box_range]
    axs[0].plot(box_range, wave_metrics_df["median_psd_dist"], marker="s",
                color=c2, linestyle="-", alpha=1, label="WaveGAN")
    axs[0].plot(box_range, stft_metrics_df["median_psd_dist"], marker="o",
                color=c3, linestyle="-", alpha=1, label="STFT-GAN")
    #axs[0].set_ylim((0, 0.8))
    axs[0].set_ylabel("Geodesic PSD Distance", fontsize=14)
    axs[0].grid(True)
    axs[0].xaxis.set_ticks_position('none')
    axs[0].xaxis.set_ticklabels([])

    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"], 'parameter_value_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_value_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    stft_dists = np.array(stft_dists)
    stft_dists = [row[~np.isnan(row)] for row in stft_dists]
    box1 = axs[1].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[1].set_ylabel(r"Estimated $\alpha$", fontsize=14)
    axs[1].grid(True)
    axs[1].xaxis.set_ticks_position('none')
    axs[1].xaxis.set_ticklabels([])
    axs[1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]],
                  ['Target', 'WaveGAN', 'STFT-GAN'], loc='upper left', fontsize=13)

    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"], 'parameter_scale_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_scale_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    stft_dists = np.array(stft_dists)
    stft_dists = [row[~np.isnan(row)] for row in stft_dists]
    box1 = axs[2].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                             boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                             capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[2].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                             boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                             capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[2].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                             boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                             capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[2].set_ylabel(r"Estimated $\gamma$", fontsize=14)
    axs[2].grid(True)
    axs[2].set_xticks(box_range)
    axs[2].set_xticklabels(noise_range, fontsize=12, rotation=45)
    axs[2].set_xlabel(r"Target $\alpha$", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    plt.savefig(os.path.join(plot_path, 'SAS_combined_plot_qt_only.eps'), dpi=300)
    plt.show()

#%%

fig, axs = plt.subplots(nrows=3, ncols=2, sharex="col", figsize = (8,8))
noise_titles = ['Feature Min-Max Data Scaling', 'Quantile Data Transformation']
noise_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

for i, (suffix, noise_title) in enumerate(zip(["", "_qt"], noise_titles)):
    model_metrics_df = metrics_df[metrics_df["noise_type"] == "SAS"]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"stftgan{suffix}"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"wavegan{suffix}"]

    box_range = list(range(len(stft_metrics_df)))
    target_range = [pos - 0.2 for pos in box_range]
    stft_range = [pos + 0.2 for pos in box_range]

    axs[0, i].plot(box_range, wave_metrics_df["median_psd_dist"], marker="s",
                   color=c2, linestyle="-", alpha=1, label="WaveGAN", linewidth=2)
    axs[0, i].plot(box_range, stft_metrics_df["median_psd_dist"], marker="o",
                   color=c3, linestyle="-", alpha=1, label="STFT-GAN", linewidth=2)
    axs[0, 0].set_ylabel("Geodesic PSD Distance", fontsize=14)
    axs[0, i].xaxis.set_ticks_position('none')
    axs[0, i].xaxis.set_ticklabels([])
    axs[0, i].grid()
    axs[0, 1].legend(loc = 'upper left', fontsize=14)
    axs[0, i].set_title(noise_title, fontsize=14)
    axs[0, i].set_ylim(0, 1.25)


    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"], 'parameter_value_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_value_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    stft_dists = np.array(stft_dists)
    stft_dists = [row[~np.isnan(row)] for row in stft_dists]
    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[1, 0].set_ylabel(r"Estimated $\alpha$", fontsize=14)
    axs[1, i].grid(True)
    axs[1, i].set_ylim(0, 3.1)
    axs[1, i].xaxis.set_ticks_position('none')
    axs[1, i].xaxis.set_ticklabels([])
    axs[1, 1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'],
                  loc='upper left', fontsize=14)

    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"], 'parameter_scale_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_scale_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    stft_dists = np.array(stft_dists)
    stft_dists = [row[~np.isnan(row)] for row in stft_dists]
    box1 = axs[2, i].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[2, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[2, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[2, 0].set_ylabel(r"Estimated $\gamma$", fontsize=14)
    axs[2, i].set_xlabel(r"Target $\alpha$", fontsize=14)
    axs[2, i].set_xticks(box_range)
    axs[2, i].set_xticklabels(noise_range, fontsize=11, rotation=45)
    axs[0, i].tick_params(axis='y', labelsize=11)
    axs[1, i].tick_params(axis='y', labelsize=11)
    axs[2, i].tick_params(axis='y', labelsize=11)
    axs[2, i].grid(True)
    axs[2,0].set_yscale('log')
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.2)
plt.savefig(os.path.join(plot_path, 'SAS_combined_plot.eps'), dpi=300)
plt.show()

#%%
# density and coverage plots

fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', figsize = (8,7))
noise_titles = ['Feature Min-Max Data Scaling', 'Quantile Data Transformation']
parameter_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

for i, (suffix, noise_title) in enumerate(zip(["", "_qt"], noise_titles)):
    model_metrics_df = metrics_df[metrics_df["noise_type"] == "SAS"]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"stftgan{suffix}"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"wavegan{suffix}"]
    CIs = np.array(list(wave_metrics_df["dtw_density_95perc_CI"]))
    yerr_wave = np.zeros((2, len(parameter_range)))
    yerr_wave[0, :] = np.array(wave_metrics_df["dtw_density"]) - CIs[:,0]
    yerr_wave[1, :] = CIs[:, 1] - np.array(wave_metrics_df["dtw_density"])
    CIs = np.array(list(stft_metrics_df["dtw_density_95perc_CI"]))
    yerr_stft = np.zeros((2, len(parameter_range)))
    yerr_stft[0, :] = np.array(stft_metrics_df["dtw_density"]) - CIs[:,0]
    yerr_stft[1, :] = CIs[:, 1] - np.array(stft_metrics_df["dtw_density"])

    axs[0, i].errorbar(box_range, wave_metrics_df["dtw_density"], yerr_wave, marker="s",
                         color=c2, linestyle="-", linewidth=2, label="WaveGAN", capsize = 5)
    axs[0, i].errorbar(box_range, stft_metrics_df["dtw_density"], yerr_stft, marker="o",
                       color=c3, linestyle="-", linewidth=2, label="STFT-GAN", capsize = 5)
    axs[0, 0].set_ylabel("DTW Density", fontsize=14)
    axs[0, i].set_title(noise_title, fontsize=14)
    axs[0, i].xaxis.set_ticks_position('none')
    axs[0, i].xaxis.set_ticklabels([])
    axs[0, i].tick_params(axis='y', labelsize=12)
    axs[0, i].grid(True)
    axs[0, 0].set_ylim(-0.1, 2.4)
    axs[0, 1].set_ylim(-0.05, 1.45)

    CIs = np.array(list(wave_metrics_df["dtw_coverage_95perc_CI"]))
    yerr_wave = np.zeros((2, len(parameter_range)))
    yerr_wave[0, :] = np.array(wave_metrics_df["dtw_coverage"]) - CIs[:,0]
    yerr_wave[1, :] = CIs[:, 1] - np.array(wave_metrics_df["dtw_coverage"])
    axs[1, i].errorbar(box_range, wave_metrics_df["dtw_coverage"], yerr_wave, marker="s",
                color=c2, linestyle="-", linewidth=2, label="WaveGAN", capsize = 5)
    CIs = np.array(list(stft_metrics_df["dtw_coverage_95perc_CI"]))
    yerr_stft = np.zeros((2, len(box_range)))
    yerr_stft[0, :] = np.array(stft_metrics_df["dtw_coverage"]) - CIs[:,0]
    yerr_stft[1, :] = CIs[:, 1] - np.array(stft_metrics_df["dtw_coverage"])
    axs[1, i].errorbar(box_range, stft_metrics_df["dtw_coverage"], yerr_stft, marker="o",
                color=c3, linestyle="-", linewidth=2, label="STFT-GAN", capsize = 5)
    axs[1, 0].set_ylabel("DTW Coverage", fontsize=14)
    axs[1, i].tick_params(axis='y', labelsize=12)
    axs[1, i].grid(True)
    axs[1, 0].set_ylim((-0.001, 0.02))
    axs[1, 1].set_ylim((-0.02, 1.02))
    axs[1, i].set_xlabel(r"Target $\alpha$", fontsize=14)
    axs[1, i].set_xticks(box_range)
    axs[1, i].set_xticklabels(parameter_range, fontsize=12, rotation=45)

    # add legend, removing the errorbars from markers
    handles, labels = axs[1, 0].get_legend_handles_labels()
    handles = [h[0] for h in handles]
    axs[1, 0].legend(handles, labels, loc='upper left', fontsize=14)

fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.2)
fig.savefig(os.path.join(plot_path, 'SAS_density_coverage.eps'), dpi=300)
fig.show()



#%%
# plot example time series and pdfs of target data

titlefont = 16
axislabelfont1 = 16
axislabelfont2 = 18
ticklabelfont1 = 14
ticklabelfont2 = 16
legendfont = 16

signal_length = 4096
t_ind = np.arange(signal_length)

# SAS_quant
alpha = [0.5, 1, 1.5]
x = np.linspace(-15,15,1000)
t_ind = np.arange(signal_length)
fig5, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fig6, axs = plt.subplots(len(alpha), 1, sharex=True, figsize=(8, 5))
linestyles = ['-', '--', '-.']

for k, a in enumerate(alpha):
    f = asn.sas_pdf(x, a)
    waveform = asn.simulate_sas_noise(signal_length, a)
    ax1.semilogy(x,f,label=fr'$\alpha$={a}', linewidth=3, linestyle=linestyles[k])
    axs[k].plot(t_ind,waveform)
    #axs[k].set_title(fr'$\alpha$ = {a}', fontsize=titlefont)
    axs[k].grid(True)
    axs[k].tick_params(labelsize=ticklabelfont1)
    axs[k].ticklabel_format(style = 'sci', scilimits = (0,0), axis='y')
ax1.legend(fontsize = legendfont)
ax1.grid(True)
ax1.set_xlabel('$x$', fontsize=axislabelfont2)
ax1.set_ylabel('Probability Density', fontsize=axislabelfont2)
ax1.tick_params(labelsize=ticklabelfont2)
fig5.tight_layout(pad=1)
fig5.savefig(os.path.join(plot_path,'sas_pdfs.eps'), dpi=300)
fig6.tight_layout(pad=3, h_pad=0.1)
fig6.supxlabel('Time Index', fontsize=axislabelfont1)
fig6.supylabel('Amplitude', fontsize=axislabelfont1)
fig6.savefig(os.path.join(plot_path, 'sas_examples.eps'), dpi=300)

#%%
# plot example target and generated waveforms

path_stft = [x for x in SAS_stft_qt_paths if re.search('alpha100/', x)]
stftpath = os.path.join(str(path_stft[0]), 'gen_distribution.h5')
path_wave = [x for x in SAS_wave_qt_paths if re.search('alpha100/', x)]
wavepath = os.path.join(str(path_wave[0]), 'gen_distribution.h5')
targetpath = "../Datasets/SAS/SAS_fixed_alpha100/test.h5"
output_path = os.path.join(plot_path, "SAS_waveform_comp_alpha100.eps")
h5f = h5py.File(stftpath, 'r')
stft_gen_data = h5f['test'][:128]
h5f = h5py.File(wavepath, 'r')
wave_gen_data = h5f['test'][:128]
h5f = h5py.File(targetpath, 'r')
targ_data = h5f['train'][:128]

num_waveforms = 3
rand_inds = np.random.randint(low=0, high=len(targ_data), size=num_waveforms)
targ_data, stft_gen_data, wave_gen_data = targ_data[rand_inds], stft_gen_data[rand_inds], wave_gen_data[rand_inds]
fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False)
for i in range(num_waveforms):
    wave_gen_waveform = wave_gen_data[i, :]
    stft_gen_waveform = stft_gen_data[i, :]
    targ_waveform = targ_data[i, :]
    axs[i, 0].plot(range(len(targ_waveform)), targ_waveform, alpha=1, linewidth=1, color=c1)
    axs[i, 1].plot(range(len(wave_gen_waveform)), wave_gen_waveform, alpha=1, linewidth=1, color=c2)
    axs[i, 2].plot(range(len(stft_gen_waveform)), stft_gen_waveform, alpha=1, linewidth=1, color=c3)
    for j in range(3):
        axs[i, j].margins(x=0, y=0.05)
        axs[2, j].set_xlabel("Time Index", fontsize=14)
        axs[i, j].grid(True)
        axs[i, j].xaxis.set_ticklabels([])
        axs[i, j].xaxis.set_ticks_position('none')
        axs[i, j].ticklabel_format(style = 'sci', scilimits = (0,0), axis='y')
    axs[i, 0].set_ylabel("Amplitude", rotation=90, fontsize=14)
axs[0, 0].set_title("Target", fontsize=14)
axs[0, 1].set_title("WaveGAN", fontsize=14)
axs[0, 2].set_title("STFT-GAN", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace=0.3)
fig.align_ylabels(axs[:,0])
fig.savefig(output_path, dpi=300)
fig.show()