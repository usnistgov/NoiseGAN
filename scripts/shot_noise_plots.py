import os
import re
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../')
import utils.shot_noise_utils as sn

separate_SNOE_SNGE_plots = False
plot_path = '../paper_plots/'
if not os.path.exists(plot_path):
   os.makedirs(plot_path)

SNGE_parent_path = "../model_results/shot/"
SNOE_parent_path = "../model_results/shot/"

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

c = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
     "#920000","#924900","#db6d00","#24ff24","#ffff6d"]
c1, c2, c3, c4 =  c[2], c[6], c[10], c[12]

#c1, c2, c3, c4 = ['#c44601', '#f57600', '#0073e6', '#054fb9']

#%%
SNGE_stft_paths = []
SNGE_wave_paths = []
for SNGE_path in SNGE_dataset_paths:
    path = os.path.join(SNGE_parent_path, SNGE_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            SNGE_wave_paths.append(model_path)
        else: # stftgan
            SNGE_stft_paths.append(model_path)

SNOE_stft_paths = []
SNOE_wave_paths = []
for SNOE_path in SNOE_dataset_paths:
    path = os.path.join(SNOE_parent_path, SNOE_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            SNOE_wave_paths.append(model_path)
        else: # stftgan
            SNOE_stft_paths.append(model_path)

test_set_groups = [SNGE_wave_paths, SNGE_stft_paths, SNOE_wave_paths, SNOE_stft_paths]
noise_set_names = ["SNGE", "SNGE", "SNOE", "SNOE"]
model_types = ["wavegan", "stftgan", "wavegan", "stftgan"]
df_temp = []
for test_set, noise_set, model in zip(test_set_groups, noise_set_names, model_types):
    for model_path in test_set:
        with open(os.path.join(model_path, "distance_metrics.json")) as f:
            data = json.load(f)
            data["config"] = model_path
            data["model_type"] = model
            data["noise_type"] = noise_set
            with open(os.path.join(model_path, "gan_train_config.json")) as f2:
                train_dict = json.load(f2)
                data['dataset'] = train_dict['dataloader_specs']['dataset_specs']['data_set']
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(os.path.join(plot_path, "shotnoise_results.csv"), index=False)

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
fig2.savefig(os.path.join(plot_path, 'sn_examples.png'))

#%%

if separate_SNOE_SNGE_plots:

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize = (5,7))
    noise_types = ["SNOE"]
    noise_titles = ["One-Sided Exponential Pulse Type"]
    parameter_range = [0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]

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
                      loc='upper left', fontsize=14)
        axs[1].set_xlabel(r"True $\nu$", fontsize=14)
        axs[1].set_ylabel(r"Estimated $\nu$", fontsize=14)
        axs[1].grid(True)
        axs[1].xaxis.set_ticks_position('none')
        axs[1].xaxis.set_ticklabels([])
        axs[1].tick_params(axis='x', which='major', labelsize=10.5)
        axs[1].tick_params(axis='y', which='major', labelsize=11)
        axs[1].set_ylim((0,5))

        axs[0].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="s",
                    color=c2, linestyle="-", alpha=1, linewidth=2, label="WaveGAN")
        axs[0].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o",
                    color=c3, linestyle="-", alpha=1, linewidth=2, label="STFT-GAN")
        axs[0].set_ylim((0, 2.3))
        axs[0].set_title(noise_title, fontsize=14)
        axs[0].set_xticks(box_range)
        axs[0].set_xticklabels(parameter_range)
        axs[0].tick_params(axis='both', which='major', labelsize=11)
        axs[0].set_ylabel("Geodesic PSD Distance", fontsize=14)
        axs[0].grid()
        axs[0].legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'SNOE_combined_plot.png'), dpi=600)
    plt.show()

    fig, axs = plt.subplots(nrows=2, sharex=True)
    fig.set_figheight(7)
    fig.set_figwidth(5)
    noise_types = ["SNGE"]
    noise_titles = ["Gaussian Pulse Type"]
    parameter_range = [0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]

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
                      loc='upper left', fontsize=14)
        axs[1].set_xlabel(r"True $\nu$", fontsize=14)
        axs[1].set_ylabel(r"Estimated $\nu$", fontsize=14)
        axs[1].grid(True)
        axs[1].xaxis.set_ticks_position('none')
        axs[1].xaxis.set_ticklabels([])
        axs[1].tick_params(axis='x', which='major', labelsize=10.5)
        axs[1].tick_params(axis='y', which='major', labelsize=11)
        axs[1].set_ylim((0,5))

        axs[0].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="s",
                    color=c2, linestyle="-", alpha=1, linewidth=2, label="WaveGAN")
        axs[0].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o",
                    color=c3, linestyle="-", alpha=1,linewidth=2, label="STFT-GAN")
        axs[0].set_ylim((0, 2.3))
        axs[0].set_title(noise_title, fontsize=14)
        axs[0].set_xticks(box_range)
        axs[0].set_xticklabels(parameter_range)
        axs[0].tick_params(axis='both', which='major', labelsize=11)
        axs[0].set_ylabel("Geodesic PSD Distance", fontsize=14)
        axs[0].grid()
        axs[0].legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'SNGE_combined_plot.png'), dpi=600)
    plt.show()

#%%

fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey = 'row', figsize = (8,7))
noise_types = ["SNOE", "SNGE"]
noise_titles = ["One-Sided Exponential Pulse Type", "Gaussian Pulse Type"]
parameter_range = [0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]

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
    axs[0, i].set_ylim((0, 2.2))
    axs[0, i].tick_params(axis='y', labelsize=12)
    axs[0, i].grid(True)
    axs[0, 0].legend(fontsize=14)

    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=wave_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[1, 0].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'],
                  loc='upper left', fontsize=14)
    axs[1, i].set_xlabel(r"True $\nu$", fontsize=14)
    axs[1, 0].set_ylabel(r"Estimated $\nu$", fontsize=14)
    axs[1, i].grid(True)
    axs[1, i].set_xticks(box_range)
    axs[1, i].set_xticklabels(parameter_range, fontsize=12, rotation=45)
    axs[1, i].tick_params(axis='y', labelsize=12)
    axs[1, i].set_ylim((0,5))

fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.1)
fig.savefig(os.path.join(plot_path, 'shot_combined_plot.png'), dpi=600)
fig.show()


#%%
# compare PSDs of target and generated shot noise with event rate = 1

path_stft = [x for x in SNOE_stft_paths if re.search('ER100', x)]
stftpath = str(path_stft[0])
path_wave = [x for x in SNOE_wave_paths if re.search('ER100', x)]
wavepath = str(path_wave[0])

fig, axs = plt.subplots(2, figsize=(5, 7), sharex=True, gridspec_kw={'hspace': 0.05})
h5f = h5py.File(os.path.join(stftpath, 'median_psds.h5'), 'r')
stft_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(os.path.join(wavepath, 'median_psds.h5'), 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()

w = np.linspace(0, 0.5, len(stft_gen_median_psd))
axs[0].plot(w, 10 * np.log10(targ_median_psd), color=c1, alpha=1, linewidth=2, label='Target')
axs[0].plot(w, 10 * np.log10(wave_gen_median_psd), color=c2, alpha=1, linewidth=2, label='WaveGAN')
axs[0].plot(w, 10 * np.log10(stft_gen_median_psd), color=c3, alpha=1, linewidth=2, label='STFT-GAN')
axs[0].set_ylabel('Power Density (dB)', fontsize=14)
axs[0].grid()
axs[0].set_xlim(-0.005, 0.5)
axs[0].tick_params(axis='both', which='major', labelsize=12)

path_stft = [x for x in SNGE_stft_paths if re.search('ER100', x)]
stftpath = str(path_stft[0])
path_wave = [x for x in SNGE_wave_paths if re.search('ER100', x)]
wavepath = str(path_wave[0])

h5f = h5py.File(os.path.join(stftpath, 'median_psds.h5'), 'r')
stft_gen_median_psd = h5f['gen'][:]
targ_median_psd = h5f['targ'][:]
h5f.close()
h5f = h5py.File(os.path.join(wavepath, 'median_psds.h5'), 'r')
wave_gen_median_psd = h5f['gen'][:]
h5f.close()

axs[1].plot(w, 10 * np.log10(targ_median_psd), color=c1, alpha=1, linewidth=2, label='Target')
axs[1].plot(w, 10 * np.log10(wave_gen_median_psd), color=c2, alpha=1, linewidth=2, label='WaveGAN')
axs[1].plot(w, 10 * np.log10(stft_gen_median_psd), color=c3, alpha=1, linewidth=2, label='STFT-GAN')
axs[1].set_ylabel('Power Density (dB)', fontsize=14)
axs[1].set_xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=14)
axs[1].grid()
axs[1].set_xlim(-.005, 0.5)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].legend(fontsize=14)

plt.subplots_adjust(left=0.16)
plt.savefig(os.path.join(plot_path, "shot_noise_psd_comparison_ER100.png"), dpi=600)
plt.show()


#%%
# plot example target and generated waveforms

path_stft = [x for x in SNOE_stft_paths if re.search('ER25/', x)]
stftpath = os.path.join(str(path_stft[0]), 'gen_distribution.h5')
path_wave = [x for x in SNOE_wave_paths if re.search('ER25/', x)]
wavepath = os.path.join(str(path_wave[0]), 'gen_distribution.h5')
targetpath = "../Datasets/shot/shot_one_sided_exponential_exponential_fixed_ER25/test.h5"
output_path = os.path.join(plot_path, "SNOE_waveform_comp_ER25.png")
h5f = h5py.File(stftpath, 'r')
stft_gen_data = h5f['test'][:128]
h5f = h5py.File(wavepath, 'r')
wave_gen_data = h5f['test'][:128]
h5f = h5py.File(targetpath, 'r')
targ_data = h5f['train'][:128]

num_waveforms = 3
rand_inds = np.random.randint(low=0, high=len(targ_data), size=num_waveforms)
targ_data, stft_gen_data, wave_gen_data = targ_data[rand_inds], stft_gen_data[rand_inds], wave_gen_data[rand_inds]
fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=True)
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
fig.subplots_adjust(hspace=0.2, wspace=0.3)
fig.align_ylabels(axs[:,0])
fig.savefig(output_path, dpi=600)
fig.show()
