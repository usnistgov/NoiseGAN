import os
import re
import json
import h5py
import sys
sys.path.insert(0,'../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.fractional_noise_utils as fn

plot_path = "../paper_plots/"
FGN_parent_path = "../model_results/FGN/"
FBM_parent_path = "../model_results/FBM/"
FGN_dataset_paths = ["FGN_fixed_H5/", "FGN_fixed_H10/", "FGN_fixed_H20/", "FGN_fixed_H30/", "FGN_fixed_H40/", "FGN_fixed_H50/",
                     "FGN_fixed_H60/", "FGN_fixed_H70/", "FGN_fixed_H80/", "FGN_fixed_H90/", "FGN_fixed_H95/"]
FBM_dataset_paths = ["FBM_fixed_H5/", "FBM_fixed_H10/", "FBM_fixed_H20/", "FBM_fixed_H30/", "FBM_fixed_H40/",
                     "FBM_fixed_H50/", "FBM_fixed_H60/", "FBM_fixed_H70/", "FBM_fixed_H80/", "FBM_fixed_H90/", "FBM_fixed_H95/"]

if not os.path.exists(plot_path):
   os.makedirs(plot_path)

c0, c1, c2, c3, c4 = "#000000", "#0072B2", "#009E73", "#D55E00", "#CC79A7"

#%%
FGN_stft_paths = []
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
        else: # stftgan
            FGN_stft_paths.append(model_path)

FBM_stft128_paths = []
FBM_stft256_paths = []
FBM_wave_paths = []
for FBM_path in FBM_dataset_paths:
    path = os.path.join(FBM_parent_path, FBM_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            FBM_wave_paths.append(model_path)
        else: # stftgan
            if train_specs_dict['dataloader_specs']['dataset_specs']['nperseg'] == 128:
                FBM_stft128_paths.append(model_path)
            elif train_specs_dict['dataloader_specs']['dataset_specs']['nperseg'] == 256:
                FBM_stft256_paths.append(model_path)

test_set_groups = [FGN_wave_paths, FGN_stft_paths, FBM_wave_paths,
                   FBM_stft128_paths, FBM_stft256_paths]
noise_set_names = ["FGN", "FGN", "FBM", "FBM", "FBM"]
model_types = ["wavegan", "stftgan", "wavegan", "stftgan", "stftgan2"]

df_temp = []
for test_set, noise_set, model in zip(test_set_groups, noise_set_names, model_types):
    for model_path in test_set:
        with open(os.path.join(model_path, "distance_metrics.json")) as f:
            data = json.load(f)
            data["config"] = model_path
            data["dataset"] = "/".join(data["config"].split("/")[3:5])
            data["model_type"] = model
            #print(f"{data['dataset']}, {model}, {noise_set}")
            data["noise_type"] = noise_set
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(plot_path + "powerlaw_results.csv", index=False)


#%%

noise_types = ["FGN"]
x_labels = [r"$H$"]
noise_titles = ["Fractional Gaussian Noise"]
ranges = {"FGN": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]}
fig, axs = plt.subplots(nrows=2, sharex=True)
fig.set_figheight(7)
fig.set_figwidth(5)

for i, (noise_type, x_label, noise_title) in enumerate(zip(noise_types, x_labels, noise_titles)):
    parameter_range = ranges[noise_type]
    model_metrics_df = metrics_df[metrics_df["noise_type"] == noise_type]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan"]
    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"],'parameter_value_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_value_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])


    box_range = list(range(len(stft_dists)))
    target_range = [pos - 0.2 for pos in box_range]
    wave_range = box_range
    stft_range = [pos + 0.2 for pos in box_range]

    box1 = axs[1].boxplot(wave_dists, showfliers=False, positions=wave_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'],
                  loc='upper left', fontsize=12)
    axs[1].set_xlabel(fr"True {x_label}", fontsize=14)
    axs[1].set_ylabel(fr"Estimated {x_label}", fontsize=14)
    axs[1].grid(True)
    axs[1].xaxis.set_ticks_position('none')
    axs[1].xaxis.set_ticklabels([])
    axs[1].tick_params(axis='both', which='major', labelsize=11)
    axs[1].set_ylim((0,1.05))

    axs[0].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="s", color=c2, linestyle="-", alpha=1,
                label="WaveGAN", linewidth=2)
    axs[0].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o", color=c3, linestyle="-", alpha=1,
                label="STFT-GAN", linewidth=2)
    axs[0].set_ylim((0, 0.5))
    axs[0].set_title(noise_title, fontsize=14)
    axs[0].set_xticks(box_range)
    axs[0].set_xticklabels(parameter_range)
    axs[0].tick_params(axis='both', which='major', labelsize=11)
    axs[0].set_ylabel("Geodesic PSD Distance", fontsize=14)
    axs[0].grid()
    axs[0].legend(loc = 'upper center', fontsize=12)

plt.tight_layout()
#plt.subplots_adjust(hspace=0.05, wspace=0.15)
plt.savefig(os.path.join(plot_path, 'FGN_combined_plot.png'), dpi=600)
plt.show()

#%%
fig, axs = plt.subplots(nrows=2, sharex=True)
fig.set_figheight(7)
fig.set_figwidth(5)
parameter_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
model_metrics_df = metrics_df[metrics_df["noise_type"] == "FBM"]
stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan"]
stft2_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan2"]
stft_dists, stft2_dists, target_dists, wave_dists = [], [], [], []
for _, stft_run in stft_metrics_df.iterrows():
    f = open(os.path.join(stft_run["config"], 'parameter_value_distributions.json'), "r")
    stft_param_dists = json.loads(f.read())
    target_dists.append(stft_param_dists["target"])
    stft_dists.append(stft_param_dists["generated"])
for _, wave_run in wave_metrics_df.iterrows():
    f = open(os.path.join(wave_run["config"], 'parameter_value_distributions.json'), "r")
    wave_param_dists = json.loads(f.read())
    wave_dists.append(wave_param_dists["generated"])
for _, stft2_run in stft2_metrics_df.iterrows():
    f = open(os.path.join(stft2_run["config"], 'parameter_value_distributions.json'), "r")
    stft_param_dists = json.loads(f.read())
    stft2_dists.append(stft_param_dists["generated"])

box_range = list(range(len(stft_dists)))
target_range = [pos - 0.3 for pos in box_range]
wave_range = [pos - 0.1 for pos in box_range]
stft_range = [pos + 0.1 for pos in box_range]
stft2_range = [pos + 0.3 for pos in box_range]
box1 = axs[1].boxplot(wave_dists, showfliers=False, positions=wave_range, widths=0.2, notch=True, patch_artist=True,
                      boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                      capprops=dict(color="black"), whiskerprops=dict(color="black"))
box2 = axs[1].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                      boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                      capprops=dict(color="black"), whiskerprops=dict(color="black"))
box3 = axs[1].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                      boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                      capprops=dict(color="black"), whiskerprops=dict(color="black"))
box4 = axs[1].boxplot(stft2_dists, showfliers=False, positions=stft2_range, widths=0.2, notch=True, patch_artist=True,
                         boxprops=dict(facecolor=c4, color=c4, alpha=1), medianprops=dict(color='black'),
                         capprops=dict(color="black"), whiskerprops=dict(color="black"))
axs[1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0], box4["boxes"][0]],
                 ['Target', 'WaveGAN', 'STFT-GAN (65x65)', 'STFT-GAN (129x65)'],
                 loc='upper left', fontsize=12)
axs[1].set_ylabel("Estimated $H$", fontsize=14)
axs[1].grid(True)
axs[1].xaxis.set_ticks_position('none')
axs[1].xaxis.set_ticklabels([])
axs[1].set_yticks(np.arange(-0.25,2, 0.25))
axs[1].tick_params(axis='both', which='major', labelsize=11)
axs[0].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="s", color=c2, linestyle="-", alpha=1,
         label="WaveGAN", linewidth=2)
axs[0].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o", color=c3, linestyle="-", alpha=1,
               label='STFT-GAN (65x65)', linewidth=2)
axs[0].plot(box_range, stft2_metrics_df["geodesic_psd_dist"], marker="^", color=c4, linestyle="-", alpha=1,
               label="STFT-GAN (129x65)", linewidth=2)
axs[0].set_title("Fractional Brownian Motion", fontsize=14)
axs[0].set_xticks(box_range)
axs[0].set_xticklabels(parameter_range)
axs[0].tick_params(axis='both', which='major', labelsize=11)
axs[0].set_ylabel("Geodesic PSD Distance", fontsize=14)
axs[1].set_xlabel("True $H$", fontsize=14)
axs[0].grid()
axs[0].legend(fontsize=12)
plt.tight_layout()
#plt.subplots_adjust(hspace=0.05, wspace=0.15)
plt.savefig(os.path.join(plot_path, 'FBM_combined_plot.png'), dpi=600)
plt.show()

#%%

fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', figsize = (8,7))
noise_types = ['FGN', 'FBM']
noise_titles = ['Fractional Gaussian Noise', 'Fractional Brownian Motion']
parameter_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

for i, (noise_type, noise_title) in enumerate(zip(noise_types, noise_titles)):
    model_metrics_df = metrics_df[metrics_df["noise_type"] == noise_type]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan"]
    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"],'parameter_value_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_value_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    if noise_type == 'FBM':
            stft2_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan2"]
            stft2_dists = []
            for _, stft2_run in stft2_metrics_df.iterrows():
                f = open(os.path.join(stft2_run["config"], 'parameter_value_distributions.json'), "r")
                stft_param_dists = json.loads(f.read())
                stft2_dists.append(stft_param_dists["generated"])

    box_range = list(range(len(stft_dists)))
    target_range = [pos - 0.2 for pos in box_range]
    wave_range = box_range
    stft_range = [pos + 0.2 for pos in box_range]

    axs[0, i].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="s",
                color=c2, linestyle="-", alpha=1, linewidth=2, label="WaveGAN")
    axs[0, i].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o",
                color=c3, linestyle="-", alpha=1, linewidth=2, label="STFT-GAN")
    axs[0, 0].set_ylabel("Geodesic PSD Distance", fontsize=14)
    axs[0, i].set_title(noise_title, fontsize=14)
    axs[0, i].xaxis.set_ticks_position('none')
    axs[0, i].xaxis.set_ticklabels([])
    axs[0, 0].set_ylim((0, 0.5))
    axs[0, 1].set_ylim((0, 1.75))
    axs[0, i].tick_params(axis='y', labelsize=12)
    axs[0, i].grid(True)

    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=wave_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    if noise_type == 'FBM':
        stft2_range = [pos + 0.3 for pos in box_range]
        axs[0, 1].plot(box_range, stft2_metrics_df["geodesic_psd_dist"], marker="^", color=c4, linestyle="-", alpha=1,
                       label="STFT-GAN (129x65)", linewidth=2)
        box4 = axs[1, 1].boxplot(stft2_dists, showfliers=False, positions=stft2_range, widths=0.2, notch=True, patch_artist=True,
                              boxprops=dict(facecolor=c4, color=c4, alpha=1), medianprops=dict(color='black'),
                              capprops=dict(color="black"), whiskerprops=dict(color="black"))
        axs[0, 1].legend(loc='upper left', fontsize=11)
        axs[1, 1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0], box4["boxes"][0]],
                      ['Target', 'WaveGAN', 'STFT-GAN (65x65)', 'STFT-GAN (129x65)'],
                      loc='upper left', fontsize=11)
    #else:
    #    axs[1, 0].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'],
    #                     loc='upper left', fontsize=14)
    axs[1, i].set_xlabel("True $H$", fontsize=14)
    axs[1, 0].set_ylabel("Estimated $H$", fontsize=14)
    axs[1, i].grid(True)
    axs[1, i].set_xticks(box_range)
    axs[1, i].set_xticklabels(parameter_range, fontsize=12, rotation=45)
    axs[1, i].tick_params(axis='y', labelsize=12)
    axs[1, 0].set_ylim((0,1.02))
    axs[1, 1].set_ylim((-0.25, 2))

fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.2)
fig.savefig(os.path.join(plot_path, 'FGN_FBM_combined_plot.png'), dpi=600)
fig.show()




#%%
# example FBM time series

titlefont = 16
axislabelfont1 = 16
axislabelfont2 = 18
ticklabelfont1 = 14
ticklabelfont2 = 16
legendfont = 16
signal_length = 4096
t_ind = np.arange(signal_length)

Hurst_index = [0.2, .5, .8]
fig, axs = plt.subplots(len(Hurst_index), 1, sharex=True, figsize=(8, 5))

for k, H in enumerate(Hurst_index):
    FGN_time_series, Gh2 = fn.simulate_FGN(signal_length, H, sigma_sq=1)
    FBM_time_series = fn.FGN_to_FBM(FGN_time_series)
    axs[k].plot(t_ind,FBM_time_series)
    #axs[k].set_title(f'H={H}', fontsize=titlefont)
    axs[k].grid(True)
    axs[k].tick_params(labelsize=ticklabelfont1)
    axs[k].set_xlim(0,4095)
fig.tight_layout(pad=3, h_pad=0.1)
fig.supxlabel('Time Index', fontsize=axislabelfont1)
fig.supylabel('Amplitude', fontsize=axislabelfont1)
fig.savefig(os.path.join(plot_path, 'fbm_examples.png'), dpi=300)

#%%
# compare PSDs of target and generated FBM noise with H=0.9

path_stft128 = [x for x in FBM_stft128_paths if re.search('H90', x)]
stft128path = str(path_stft128[0])
path_stft256 = [x for x in FBM_stft256_paths if re.search('H90', x)]
stft256path = str(path_stft256[0])
path_wave = [x for x in FBM_wave_paths if re.search('H90', x)]
wavepath = str(path_wave[0])

fig, ax = plt.subplots(1, figsize=(6, 4))
h5f = h5py.File(os.path.join(stft128path, 'median_psds.h5'), 'r')
stft128_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(os.path.join(stft256path, 'median_psds.h5'), 'r')
stft256_gen_median_psd = h5f['gen'][:]
h5f.close()
h5f = h5py.File(os.path.join(wavepath, 'median_psds.h5'), 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()

w = np.linspace(0, 0.5, len(stft128_gen_median_psd))
ax.plot(w, 10 * np.log10(targ_median_psd), color=c1, alpha=1, linewidth=2, label='Target')
ax.plot(w, 10 * np.log10(wave_gen_median_psd), color=c2, alpha=1, linewidth=2, label='WaveGAN')
ax.plot(w, 10 * np.log10(stft128_gen_median_psd), color=c3, alpha=1, linewidth=2, label=r'STFT-GAN (65 $\times$ 65)')
ax.plot(w, 10 * np.log10(stft256_gen_median_psd), color=c4, alpha=1, linewidth=2, label=r'STFT-GAN (129 $\times$ 65)')
ax.set_ylim([-20, 60])
ax.set_ylabel('Power Density (dB)', fontsize=14)
ax.set_xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid()
fig.tight_layout(pad=1)
ax.legend(fontsize=14)
fig.savefig(os.path.join(plot_path, "FBM_H90_psd_comparison.png"), dpi=600)
fig.show()

#%%
# plot example target and generated waveforms for FBM

path_stft256 = [x for x in FBM_stft256_paths if re.search('H50', x)]
stft256path = str(path_stft256[0])
path_wave = [x for x in FBM_wave_paths if re.search('H50', x)]
wavepath = str(path_wave[0])
stftpath_gen = os.path.join(stft256path, 'gen_distribution.h5')
wavepath_gen = os.path.join(wavepath, 'gen_distribution.h5')
targetpath = "../Datasets/FBM/FBM_fixed_H50/test.h5"
h5f = h5py.File(stftpath_gen, 'r')
stft_gen_data = h5f['test'][:128]
h5f = h5py.File(wavepath_gen, 'r')
wave_gen_data = h5f['test'][:128]
h5f = h5py.File(targetpath, 'r')
targ_data = h5f['train'][:128]

num_waveforms = 3
rand_inds = np.random.randint(low=0, high=len(targ_data), size=num_waveforms)
targ_data, stft_gen_data, wave_gen_data = targ_data[rand_inds], stft_gen_data[rand_inds], wave_gen_data[rand_inds]
fig, axs = plt.subplots(nrows=3, ncols=3, sharex = True, sharey = True)
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
    axs[i, 0].set_ylabel("Amplitude", rotation=90, fontsize=14)
axs[0, 0].set_title("Target", fontsize=14)
axs[0, 1].set_title("WaveGAN", fontsize=14)
axs[0, 2].set_title("STFT-GAN", fontsize=14)
fig.tight_layout()
#fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.align_ylabels(axs[:,0])
output_path = os.path.join(plot_path, "FBM_waveform_comp_H50.png")
plt.savefig(output_path, dpi=600)
plt.show()