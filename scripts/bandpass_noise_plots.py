#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots for bandpass noise
"""

import os
import re
import json
import h5py
import os.path
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from labellines import labelLines
#plt.style.use('seaborn-colorblind')

c = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
     "#920000","#924900","#db6d00","#24ff24","#ffff6d"]
c1, c2, c3, c4 =  c[2], c[6], c[10], c[12]

#%%


plot_path = "../paper_plots/"
if not os.path.exists(plot_path):
   os.makedirs(plot_path)

BPWN_parent_path = "/data/noise-gan/model_results/bandpass/"
BPWN_dataset_paths = ["band0/", "band1/", "band2/", "band3/", "band4/", "band5/", "band6/", "band7/"]

BPWN_stft_paths = []
BPWN_wave_paths = []
for BPWN_path in BPWN_dataset_paths:
    path = os.path.join(BPWN_parent_path, BPWN_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            BPWN_wave_paths.append(model_path)
        else: # stftgan
            BPWN_stft_paths.append(model_path)

test_set_groups = [BPWN_wave_paths, BPWN_stft_paths]
model_types = ["wavegan", "stftgan"]
df_temp = []
for test_set, model in zip(test_set_groups, model_types):
    for model_path in test_set:
        with open(os.path.join(model_path, "summary_metrics.json")) as f:
            data = json.load(f)
            data["config"] = model_path
            data["model_type"] = model
            data["noise_type"] = "BPWN"
            with open(os.path.join(model_path, "gan_train_config.json")) as f2:
                train_dict = json.load(f2)
                data['dataset'] = train_dict['dataloader_specs']['dataset_specs']['data_set']
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(os.path.join(plot_path, "BPWN_results.csv"), index=False)

#%%

fig = plt.figure(figsize=(5, 4))
band_nums = [0, 1, 2, 3, 4, 5, 6, 7]
model_metrics_df = metrics_df[metrics_df["noise_type"] == "BPWN"]
stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan"]
plt.plot(band_nums, wave_metrics_df["median_psd_dist"], marker="s", color=c2,
         linestyle="-", alpha=1, label="WaveGAN", linewidth=2)
plt.plot(band_nums, stft_metrics_df["median_psd_dist"], marker="o", color=c3,
         linestyle="-", alpha=1, label="STFT-GAN", linewidth=2)
plt.ylabel("Geodesic PSD Distance", fontsize=14)
plt.xticks(band_nums)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Band Number", fontsize=14)
plt.ylim((0,3.2))
plt.grid()
plt.legend(loc = 'center right', fontsize=14)
fig.tight_layout()
plt.savefig(os.path.join(plot_path, "BPWN_psd_distance.eps"), dpi=300)
plt.show()

#%%

path_stft = [x for x in BPWN_stft_paths if re.search('band3', x)]
stftpath = str(path_stft[0])
path_wave = [x for x in BPWN_wave_paths if re.search('band3', x)]
wavepath = str(path_wave[0])
h5f = h5py.File(os.path.join(stftpath, 'median_psds.h5'), 'r')
stft_gen_median_psd = h5f['gen_median_psd'][:]
targ_median_psd = h5f['targ_median_psd'][:]
h5f.close()
h5f = h5py.File(os.path.join(wavepath, 'median_psds.h5'), 'r')
wave_gen_median_psd = h5f['gen_median_psd'][:]
h5f.close()

w = np.linspace(0, 0.5, len(stft_gen_median_psd))
fig = plt.figure(figsize=(5, 4))
plt.plot(w, 10 * np.log10(targ_median_psd), color=c1, alpha=1, linewidth=2, label='Target')
plt.plot(w, 10 * np.log10(wave_gen_median_psd), color=c2, linewidth=2, alpha=1, label='WaveGAN')
plt.plot(w, 10 * np.log10(stft_gen_median_psd), color=c3, linewidth=2, alpha=1, label='STFT-GAN')
plt.ylabel('Power Density (dB)', fontsize=14)
plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.xlim(0, 0.5)
fig.tight_layout()
plt.legend(fontsize=14)
plt.savefig(os.path.join(plot_path, "BPWN_band3_psd_comparison.eps"), dpi=300)
plt.show()

#%%

titlefont = 16
axislabelfont1 = 16
axislabelfont2 = 18
ticklabelfont1 = 14
ticklabelfont2 = 16
legendfont = 16

num_bands = 8
sampling_frequency = 2  # Nyquist freq is 1 (normalized digital frequency)
filter_order = 40
half_bandwidth = 1.0 / (2 * num_bands + (num_bands - 1))
full_bandwidth = 2.0 * half_bandwidth
sos_coeffs = {}
fig3, ax1 = plt.subplots(1, 1, figsize=(7, 4))
# design band-pass filters and plot frequency responses
for num in range(num_bands):
    if num == 0:
        cutoff_freqs = [full_bandwidth]
        filter_type = "lowpass"
    elif num == num_bands - 1:
        cutoff_freqs = [full_bandwidth + (half_bandwidth*num)
                            + (full_bandwidth*(num-1))]
        filter_type = "highpass"
    else:
        cutoff_freqs = [full_bandwidth + (half_bandwidth*num)
                           + (full_bandwidth*(num-1)), full_bandwidth
                           + (half_bandwidth*num) + (full_bandwidth*num)]
        filter_type = "bandpass"

    # design IIR filter
    sos = signal.iirfilter(N=filter_order, Wn=cutoff_freqs, btype=filter_type, ftype='butter',
                              fs=sampling_frequency, output='sos')
    sos_coeffs[num] = sos
    w, h = signal.sosfreqz(sos)
    ax1.plot(w/(2*np.pi), 20*np.log10(np.abs(h)), linewidth=3, label = str(num), color = '#3b3b3b')
labelLines(ax1.get_lines(), xvals = np.arange(0.02, 0.5, .066), fontsize=14)
ax1.grid(True)
ax1.set_xlim([0,0.5])
ax1.set_ylim([-80, 5])
ax1.set_xlabel('Normalized Digitial Frequency (cycles/sample)', fontsize=axislabelfont2)
ax1.set_ylabel('Filter Gain (dB)', fontsize=axislabelfont2)
ax1.tick_params(labelsize=ticklabelfont2)
fig3.tight_layout(pad=1)
fig3.savefig(os.path.join(plot_path, 'BP_filters.eps'), dpi=300)

#%%
# density and coverage plots


fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (5, 4))
band_nums = [0, 1, 2, 3, 4, 5, 6, 7]
CIs = np.array(list(wave_metrics_df["dtw_density_95perc_CI"]))
yerr_wave = np.zeros((2, len(band_nums)))
yerr_wave[0, :] = np.array(wave_metrics_df["dtw_density"]) - CIs[:,0]
yerr_wave[1, :] = CIs[:, 1] - np.array(wave_metrics_df["dtw_density"])
ax1.errorbar(band_nums, wave_metrics_df["dtw_density"], yerr_wave, marker="s",
            color=c2, linestyle="-", linewidth=2, label="WaveGAN", capsize = 5)
CIs = np.array(list(stft_metrics_df["dtw_density_95perc_CI"]))
yerr_stft = np.zeros((2, len(band_nums)))
yerr_stft[0, :] = np.array(stft_metrics_df["dtw_density"]) - CIs[:,0]
yerr_stft[1, :] = CIs[:, 1] - np.array(stft_metrics_df["dtw_density"])
ax1.errorbar(band_nums, stft_metrics_df["dtw_density"], yerr_stft, marker="o",
            color=c3, linestyle="-", linewidth=2, label="STFT-GAN", capsize = 5)
ax1.set_ylabel("DTW Density", fontsize=14)
ax1.set_ylim((0, 2))
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True)
ax1.set_xlabel("Band Number", fontsize=14)
ax1.set_xticks(band_nums)

# add legend, removing errorbars from markers
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.legend(handles, labels, loc='upper right', fontsize=14)

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize = (5, 4))
CIs = np.array(list(wave_metrics_df["dtw_coverage_95perc_CI"]))
yerr_wave = np.zeros((2, len(band_nums)))
yerr_wave[0, :] = np.array(wave_metrics_df["dtw_coverage"]) - CIs[:,0]
yerr_wave[1, :] = CIs[:, 1] - np.array(wave_metrics_df["dtw_coverage"])
ax2.errorbar(band_nums, wave_metrics_df["dtw_coverage"], yerr_wave, marker="s",
            color=c2, linestyle="-", linewidth=2, label="WaveGAN", capsize = 5)
CIs = np.array(list(stft_metrics_df["dtw_coverage_95perc_CI"]))
yerr_stft = np.zeros((2, len(band_nums)))
yerr_stft[0, :] = np.array(stft_metrics_df["dtw_coverage"]) - CIs[:,0]
yerr_stft[1, :] = CIs[:, 1] - np.array(stft_metrics_df["dtw_coverage"])
ax2.errorbar(band_nums, stft_metrics_df["dtw_coverage"], yerr_stft, marker="o",
            color=c3, linestyle="-", linewidth=2, label="STFT-GAN", capsize = 5)

ax2.set_ylabel("DTW Coverage", fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True)
ax2.set_xlabel("Band Number", fontsize=14)
ax2.set_xticks(band_nums)

fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(os.path.join(plot_path, 'BPWN_density.eps'), dpi=300)
fig2.savefig(os.path.join(plot_path, 'BPWN_coverage.eps'), dpi=300)
fig1.show()
fig2.show()
