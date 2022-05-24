# -*- coding: utf-8 -*-
"""
script for plotting noise waveform examples, PDFs, and PSDs
"""

import os
import json
import h5py
import os.path
import numpy as np
import pandas as pd
from spectrum import pmtm
import scipy.signal as signal
import matplotlib.pyplot as plt
import utils.bg_noise_utils as bg
import utils.shot_noise_utils as sn
import utils.fractional_noise_utils as fn
import utils.alpha_stable_noise_utils as asn


def get_median_psd(path):
    print(path)
    if os.path.exists(f"./Datasets/{path}/median_psd.h5"):
        h5f = h5py.File(f"./Datasets/{path}/median_psd.h5", 'r')
        targ_median_psd = h5f['median_psd'][:]
        h5f.close()
    else:
        h5f = h5py.File(f"./Datasets/{path}/test.h5", 'r')
        targ_dataset = h5f['train'][:]
        h5f.close()
        targ_data = np.array(targ_dataset[:, 1:]).astype(float)
        targ_PSD_estimates, gen_PSD_estimates = [], []
        for i, targ_sample in enumerate(targ_data):
            Sk, weights, _eigenvalues = pmtm(targ_sample, NW=4, k=7, method='eigen')
            Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
            Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
            targ_PSD_estimates.append(Pxx)
        targ_median_psd = np.median(np.array(targ_PSD_estimates), axis=0)
        h5f = h5py.File(f"./Datasets/{path}/median_psd.h5", 'w')
        h5f.create_dataset('median_psd', data=targ_median_psd)
        h5f.close()
    return targ_median_psd


def waveform_comparison(gen_data, wave_gen_data, targ_data, output_path, common_yscale=False, sas=False):
    num_waveforms = 3
    rand_inds = np.random.randint(low=0, high=len(targ_data), size=num_waveforms)
    targ_data, gen_data, wave_gen_data = targ_data[rand_inds], gen_data[rand_inds], wave_gen_data[rand_inds]
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False, sharey='row' if common_yscale else common_yscale)
    min_val = min(np.amin(targ_data), np.amin(gen_data), np.amin(wave_gen_data))
    max_val = max(np.amax(targ_data), np.amax(gen_data), np.amax(wave_gen_data))
    for i in range(num_waveforms):
        wave_gen_waveform = wave_gen_data[i, :]
        gen_waveform = gen_data[i, :]
        targ_waveform = targ_data[i, :]
        axs[i, 0].plot(range(len(targ_waveform)), targ_waveform, alpha=1, linewidth=1, color="green")
        axs[i, 1].plot(range(len(wave_gen_waveform)), wave_gen_waveform, alpha=1, linewidth=1, color="red")
        axs[i, 2].plot(range(len(gen_waveform)), gen_waveform, alpha=1, linewidth=1, color="blue")
        for j in range(3):
            axs[i, j].margins(x=0, y=0.05)
            axs[2, j].set_xlabel("Time")
            axs[i, j].grid(True)
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].xaxis.set_ticks_position('none')
            if sas:
                axs[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[i, 0].set_ylabel("Amplitude", rotation=90)
    axs[0, 0].set_title("Target")
    axs[0, 1].set_title("WaveGAN")
    axs[0, 2].set_title("STFT-GAN")
    plt.tight_layout()
    if common_yscale:
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
    else:
        plt.subplots_adjust(hspace=0.15, wspace=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()

#%%


plot_path = "./paper_plots/"
BPWN_parent_path = "./model_results/BPWN/"
BPWN_dataset_paths = ["noise_BPWN_bandpass_8_bands_band0/", "noise_BPWN_bandpass_8_bands_band1/",
                      "noise_BPWN_bandpass_8_bands_band2/", "noise_BPWN_bandpass_8_bands_band3/",
                      "noise_BPWN_bandpass_8_bands_band4/", "noise_BPWN_bandpass_8_bands_band5/",
                      "noise_BPWN_bandpass_8_bands_band6/", "noise_BPWN_bandpass_8_bands_band7/"]
BPWN_wave_paths = [BPWN_parent_path + path + "wavegan_ps/" for path in BPWN_dataset_paths]
BPWN_stft_paths = [BPWN_parent_path + path + "stftgan/" for path in BPWN_dataset_paths]
test_set_groups = [BPWN_wave_paths, BPWN_stft_paths]
model_types = ["wavegan_ps", "stftgan"]
df_temp = []
for test_set, model in zip(test_set_groups, model_types):
    for model_path in test_set:
        with open(model_path + "distance_metrics.json") as f:
            data = json.load(f)
            data["config"] = model_path
            data["model_type"] = model
            data["noise_type"] = "BPWN"
            with open(model_path + "gan_train_config.json") as f2:
                train_dict = json.load(f2)
                data['dataset'] = train_dict['dataloader_specs']['dataset_specs']['data_set']
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(plot_path + "BPWN_results.csv", index=False)

fig = plt.figure(figsize=(6, 4))
band_nums = [0, 1, 2, 3, 4, 5, 6, 7]
model_metrics_df = metrics_df[metrics_df["noise_type"] == "BPWN"]
stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan_ps"]
plt.plot(band_nums, stft_metrics_df["geodesic_psd_dist"], marker="o", color="blue",
         linestyle="-", alpha=0.7, label="STFTGAN")
plt.plot(band_nums, wave_metrics_df["geodesic_psd_dist"], marker="o", color="red",
         linestyle="-", alpha=0.7, label="WaveGAN")
plt.ylabel(r"$d_g$", fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel("Band Number", fontsize=12)
plt.grid()
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(plot_path + "/BPWN_psd_plots.png", dpi=300)
plt.show()


#%%
stftpath = "./model_results/shot_noise/SNOE/noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER25/stftgan/gen_distribution.h5"
wavepath = "./model_results/shot_noise/SNOE/noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER25/wavegan_ps/gen_distribution.h5"
targetpath = "./Datasets/SNOE/shot_one_sided_exponential_exponential_fixed_ER25/test.h5"
output_path = "./paper_plots/SNOE_waveform_comp.png"
h5f = h5py.File(stftpath, 'r')
stft_gen_data = h5f['test'][:128]
h5f = h5py.File(wavepath, 'r')
wave_gen_data = h5f['test'][:128]
h5f = h5py.File(targetpath, 'r')
targ_data = h5f['train'][:128]
waveform_comparison(stft_gen_data, wave_gen_data, targ_data, output_path, common_yscale=True)



#%%
stftpath = "./model_results/impulsive_noise/BGN_quant/BGN_BG_fixed_IP5/stftgan/gen_distribution.h5"
wavepath = "./model_results/impulsive_noise/BGN_quant/BGN_BG_fixed_IP5/wavegan_ps/gen_distribution.h5"
targetpath = "./Datasets/BGN/BG_fixed_IP5/test.h5"
output_path = "./paper_plots/BGN_waveform_comp.png"
h5f = h5py.File(stftpath, 'r')
stft_gen_data = h5f['test'][:256]
h5f = h5py.File(wavepath, 'r')
wave_gen_data = h5f['test'][:256]
h5f = h5py.File(targetpath, 'r')
targ_data = h5f['train'][:256]
waveform_comparison(stft_gen_data, wave_gen_data, targ_data, output_path, common_yscale=True)


stftpath = "./model_results/impulsive_noise/SAS_quant/SAS_SAS_fixed_alpha100/stftgan/gen_distribution.h5"
wavepath = "./model_results/impulsive_noise/SAS_quant/SAS_SAS_fixed_alpha100/wavegan_ps/gen_distribution.h5"
targetpath = "./Datasets/SAS/SAS_fixed_alpha100/test.h5"
output_path = "./paper_plots/SAS_waveform_comp.png"
h5f = h5py.File(stftpath, 'r')
stft_gen_data = h5f['test'][:]
h5f = h5py.File(wavepath, 'r')
wave_gen_data = h5f['test'][:]
h5f = h5py.File(targetpath, 'r')
targ_data = h5f['train'][:]
waveform_comparison(stft_gen_data, wave_gen_data, targ_data, output_path, common_yscale=False, sas=True)


#%%

bpwn_paths = ["BPWN/bandpass_8_bands_band0/", "BPWN/bandpass_8_bands_band1/", "BPWN/bandpass_8_bands_band2/",
              "BPWN/bandpass_8_bands_band3/", "BPWN/bandpass_8_bands_band4/", "BPWN/bandpass_8_bands_band5/",
              "BPWN/bandpass_8_bands_band6/", "BPWN/bandpass_8_bands_band7/"]
snot_noise_1 = ["SNGE/shot_gaussian_exponential_fixed_ER100/", "SNLE/shot_linear_exponential_exponential_fixed_ER100/",
                "SNOE/shot_one_sided_exponential_exponential_fixed_ER100/"]
shot_noise_lables = ["Gaussian", "Linear Exponential", "One-sided Exponential"]
frac_noise_1 = ["FBM/FBM_fixed_H20/", "FGN/FGN_fixed_H20/", "FDWN/FDWN_fixed_H20/"]
frac_noise_label = ["FBM", "FGN", "FDWN"]
frac_noise_2 = ["FBM/FBM_fixed_H80/", "FGN/FGN_fixed_H80/", "FDWN/FDWN_fixed_H80/"]

# BPWN
fig = plt.figure(figsize=(8,5))
for path in bpwn_paths:
    targ_median_psd = get_median_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    plt.plot(w, 10 * np.log10(targ_median_psd), alpha=0.75, linewidth=3)
plt.title(f"")
plt.ylabel('Power Density (dB)', fontsize=16)
plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=16)
plt.grid()
plt.margins(x=0)
plt.ylim((-70, -10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"./paper_plots/BPWN_target_psd_comparison.png", dpi=300)
plt.show()

# Shot Noise
fig = plt.figure(figsize=(8,5))
for path, label in zip(snot_noise_1, shot_noise_lables):
    targ_median_psd = get_median_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    plt.plot(w, 10 * np.log10(targ_median_psd), alpha=0.75, linewidth=3, label=label)
plt.ylabel('Power Density (dB)', fontsize=16)
plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=16)
plt.grid()
plt.legend(fontsize=14)
plt.ylim((-80, 20))
plt.margins(x=0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"./paper_plots/Shot_Noise_target_psd_comparison.png", dpi=300)
plt.show()

# Power Law Noise
fig, axs = plt.subplots(2, figsize=(8, 8), sharex=True, gridspec_kw={'hspace': 0.1})
for path, label in zip(frac_noise_1, frac_noise_label):
    targ_median_psd = get_median_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    axs[0].plot(w, 10 * np.log10(targ_median_psd), alpha=0.75, label=label, linewidth=3)
axs[0].set_ylabel('Power Density (dB)', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[0].grid()
axs[0].margins(x=0)
axs[0].legend(fontsize=14)
plt.xscale("log")

for path, label in zip(frac_noise_2, frac_noise_label):
    targ_median_psd = get_median_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    axs[1].plot(w, 10 * np.log10(targ_median_psd), alpha=0.75, label=label, linewidth=3)
axs[1].set_ylabel('Power Density (dB)', fontsize=16)
plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=14)
axs[1].grid()
axs[1].margins(x=0)
plt.xscale("log")
plt.savefig(f"./paper_plots/frac_noise_target_psd_comparison.png", dpi=300)
plt.show()



#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt

stftpath = "./model_results/BPWN/noise_BPWN_bandpass_8_bands_band3/stftgan/"
wavepath = "./model_results/BPWN/noise_BPWN_bandpass_8_bands_band3/wavegan_ps/"
h5f = h5py.File(f"{stftpath}/median_psds.h5", 'r')
stft_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(f"{wavepath}/median_psds.h5", 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()

w = np.linspace(0, 0.5, len(stft_gen_median_psd))
plt.plot(w, 10 * np.log10(targ_median_psd), color='green', alpha=0.65, label=f'Target')
plt.plot(w, 10 * np.log10(wave_gen_median_psd), color='red', alpha=0.65, label=f'WaveGAN')
plt.plot(w, 10 * np.log10(stft_gen_median_psd), color='blue', alpha=0.65, label=f'STFT-GAN')
plt.ylabel('Power Density (dB)')
plt.xlabel("Normalized Digital Frequency (cycles/sample)")
plt.grid()
plt.margins(x=0)
plt.legend()
plt.savefig("./paper_plots/BPWN_band3_psd_comparison.png", dpi=300)
plt.show()


stftpath = "./model_results/power_law_noise/FBM/FBM_fixed_H80/stftgan/"
wavepath = "./model_results/power_law_noise/FBM/FBM_fixed_H80/wavegan_ps/"
stftpath2 = "./model_results/power_law_noise/FBM/FBM_fixed_H80/stftgan2/"
h5f = h5py.File(f"{stftpath}/median_psds.h5", 'r')
stft_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(f"{wavepath}/median_psds.h5", 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()
h5f = h5py.File(f"{stftpath2}/median_psds.h5", 'r')
stft2_gen_median_psd = h5f['gen'][:]
h5f.close()

w = np.linspace(0, 0.5, len(stft_gen_median_psd))
plt.plot(w, 10 * np.log10(targ_median_psd), color='green', alpha=0.65, label=f'Target')
plt.plot(w, 10 * np.log10(wave_gen_median_psd), color='red', alpha=0.65, label=f'WaveGAN')
plt.plot(w, 10 * np.log10(stft_gen_median_psd), color='blue', alpha=0.65, label=f'STFT-GAN (65x65)')
plt.plot(w, 10 * np.log10(stft2_gen_median_psd), color='purple', alpha=0.65, label=f'STFT-GAN (129x65)')
plt.ylabel('Power Density (dB)')
plt.xlabel("Normalized Digital Frequency (cycles/sample)")
plt.grid()
plt.margins(x=0)
plt.legend()
plt.savefig("./paper_plots/FBM/FBM_H80_psd_comparison.png", dpi=300)
plt.show()

#%%

fig, axs = plt.subplots(2, figsize=(5, 7), sharex=True, gridspec_kw={'hspace': 0.05})
stftpath = "./model_results/shot_noise/SNOE/noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER100/stftgan/"
wavepath = "./model_results/shot_noise/SNOE/noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER100/wavegan_ps/"
h5f = h5py.File(f"{stftpath}/median_psds.h5", 'r')
stft_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(f"{wavepath}/median_psds.h5", 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()

w = np.linspace(0, 0.5, len(stft_gen_median_psd))
axs[0].plot(w, 10 * np.log10(targ_median_psd), color='green', alpha=0.65, label=f'Target')
axs[0].plot(w, 10 * np.log10(wave_gen_median_psd), color='red', alpha=0.65, label=f'WaveGAN')
axs[0].plot(w, 10 * np.log10(stft_gen_median_psd), color='blue', alpha=0.65, label=f'STFT-GAN')
axs[0].set_ylabel('Power Density (dB)')
axs[0].grid()
axs[0].margins(x=0)


stftpath = "./model_results/shot_noise/SNGE/noise_SNGE_shot_gaussian_exponential_fixed_ER100/stftgan/"
wavepath = "./model_results/shot_noise/SNGE/noise_SNGE_shot_gaussian_exponential_fixed_ER100/wavegan_ps/"
h5f = h5py.File(f"{stftpath}/median_psds.h5", 'r')
stft_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(f"{wavepath}/median_psds.h5", 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()

axs[1].plot(w, 10 * np.log10(targ_median_psd), color='green', alpha=0.65, label=f'Target')
axs[1].plot(w, 10 * np.log10(wave_gen_median_psd), color='red', alpha=0.65, label=f'WaveGAN')
axs[1].plot(w, 10 * np.log10(stft_gen_median_psd), color='blue', alpha=0.65, label=f'STFT-GAN')
axs[1].set_ylabel('Power Density (dB)')
axs[1].set_xlabel("Normalized Digital Frequency (cycles/sample)")
axs[1].grid()
axs[1].margins(x=0)
axs[1].legend()
plt.savefig("./paper_plots/Shot_Noise_psd_comparison.png", dpi=300)
plt.show()


#%%

titlefont = 16
axislabelfont1 = 16
axislabelfont2 = 18
ticklabelfont1 = 14
ticklabelfont2 = 16
legendfont = 16

signal_length = 4096
t_ind = np.arange(signal_length)


# fractional noise --------------------
Hurst_index = [0.2, .5, .8]
fig1, axs = plt.subplots(len(Hurst_index), 1, sharex=True, figsize=(8, 5))

for k, H in enumerate(Hurst_index):
    FGN_time_series, Gh2 = fn.simulate_FGN(signal_length, H, sigma_sq=1)
    FBM_time_series = fn.FGN_to_FBM(FGN_time_series)
    #FDWN_time_series = fn.simulate_FDWN(signal_length, dfrac = H-.5)
    axs[k].plot(t_ind,FBM_time_series)
    #axs[k].set_title(f'H={H}', fontsize=titlefont)
    axs[k].grid(True)
    axs[k].tick_params(labelsize=ticklabelfont1)
fig1.tight_layout(pad=3, h_pad=0.1)
fig1.supxlabel('Time Index', fontsize=axislabelfont1)
fig1.supylabel('Amplitude', fontsize=axislabelfont1)
path = '../example_plots/'
isExist = os.path.exists(path)
if not isExist:
  os.makedirs(path)
fig1.savefig('../example_plots/fbm_examples.png')

# shot noise --------------------------
tau_d = 1  # pulse duration
beta = 1  # mean pulse amplitude
theta = 0.1  # normalized time step = delta_t/tau_d
pulse_type = 'one_sided_exponential'
amp_distrib = 'exponential'
event_rate = [0.25, 0.5, 2]
fig2, axs = plt.subplots(len(event_rate), 1, sharex=True, sharey=True, figsize=(8, 5))
for k, er in enumerate(event_rate):
    sn_time_series = sn.simulate_shot_noise(signal_length, pulse_type, amp_distrib, er, tau_d, beta, theta)
    axs[k].plot(t_ind,sn_time_series)
    #axs[k].set_title(fr'$\nu$={er}', fontsize=titlefont)
    axs[k].grid(True)
    axs[k].tick_params(labelsize=ticklabelfont1)
fig2.tight_layout(pad=3, h_pad=0.1)
fig2.supxlabel('Time Index', fontsize=axislabelfont1)
fig2.supylabel('Amplitude', fontsize=axislabelfont1)
fig2.savefig('../example_plots/sn_examples.png')

# impulse noise -----------------

# BG
x = np.linspace(-3.5,3.5,1000)
sig_w=0.1
sig_i=1
impulse_prob = [.01, .05, .1]
fig3, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fig4, axs = plt.subplots(len(impulse_prob), 1, sharex=True, sharey=True, figsize=(8, 5))

for k, p in enumerate(impulse_prob):
    f_BG = bg.bg_pdf(x, p, sig_w, sig_i)
    waveform = bg.simulate_bg_noise(signal_length, p, sig_w, sig_i)
    ax1.semilogy(x,f_BG,label=f'p={p}', linewidth=3)
    axs[k].plot(t_ind,waveform)
    #axs[k].set_title(f'p={p}', fontsize=titlefont)
    axs[k].grid(True)
    axs[k].tick_params(labelsize=ticklabelfont1)
ax1.legend(fontsize = legendfont)
ax1.grid(True)
ax1.set_xlabel('$x$', fontsize=axislabelfont2)
ax1.set_ylabel('Probability Density', fontsize=axislabelfont2)
ax1.tick_params(labelsize=ticklabelfont2)
fig3.tight_layout(pad=1)
fig3.savefig('../example_plots/bg_pdfs.png')
fig4.tight_layout(pad=3, h_pad=0.1)
fig4.supxlabel('Time Index', fontsize=axislabelfont1)
fig4.supylabel('Amplitude', fontsize=axislabelfont1)
fig4.savefig('../example_plots/bg_examples.png')

# SAS_quant
alpha = [0.5, 1, 1.5]
x = np.linspace(-15,15,1000)
t_ind = np.arange(signal_length)
fig5, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fig6, axs = plt.subplots(len(alpha), 1, sharex=True, figsize=(8, 5))

for k, a in enumerate(alpha):
    f = asn.sas_pdf(x, a)
    waveform = asn.simulate_sas_noise(signal_length, a)
    ax1.semilogy(x,f,label=fr'$\alpha$={a}', linewidth=3)
    axs[k].plot(t_ind,waveform)
    #axs[k].set_title(fr'$\alpha$ = {a}', fontsize=titlefont)
    axs[k].grid(True)
    axs[k].tick_params(labelsize=ticklabelfont1)
ax1.legend(fontsize = legendfont)
ax1.grid(True)
ax1.set_xlabel('$x$', fontsize=axislabelfont2)
ax1.set_ylabel('Probability Density', fontsize=axislabelfont2)
ax1.tick_params(labelsize=ticklabelfont2)
fig5.tight_layout(pad=1)
fig5.savefig('../example_plots/sas_pdfs.png')
fig6.tight_layout(pad=3, h_pad=0.1)
fig6.supxlabel('Time Index', fontsize=axislabelfont1)
fig6.supylabel('Amplitude', fontsize=axislabelfont1)
fig6.savefig('../example_plots/sas_examples.png')

#  band-pass filtered white noise -------------------------------

num_bands = 8
sampling_frequency = 2  # Nyquist freq is 1 (normalized digital frequency)
filter_order = 40
half_bandwidth = 1.0 / (2 * num_bands + (num_bands - 1))
full_bandwidth = 2.0 * half_bandwidth
sos_coeffs = {}
fig7, ax1 = plt.subplots(1, 1, figsize=(8, 4))
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
    ax1.plot(w/(2*np.pi), 20*np.log10(np.abs(h)), linewidth=3)
ax1.grid(True)
ax1.set_xlim([0,0.5])
ax1.set_ylim([-105, 5])
ax1.set_xlabel('Normalized Digitial Frequency (cycles/sample)', fontsize=axislabelfont2)
ax1.set_ylabel('Filter Gain (dB)', fontsize=axislabelfont2)
ax1.tick_params(labelsize=ticklabelfont2)

fig7.tight_layout(pad=1)
fig7.savefig('../example_plots/BP_filters.png')