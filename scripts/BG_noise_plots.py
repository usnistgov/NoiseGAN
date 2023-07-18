import os
import re
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.insert(0,'../')
import utils.bg_noise_utils as bg

plot_path = '../paper_plots/'
if not os.path.exists(plot_path):
   os.makedirs(plot_path)

BG_parent_path = "/data/noise-gan/model_results/BG/"

BG_dataset_paths = ["BG_fixed_IP1/", "BG_fixed_IP5/", "BG_fixed_IP10/", "BG_fixed_IP15/",
                     "BG_fixed_IP20/", "BG_fixed_IP30/", "BG_fixed_IP40/", "BG_fixed_IP50/",
                     "BG_fixed_IP60/", "BG_fixed_IP70/", "BG_fixed_IP80/", "BG_fixed_IP90/"]

c = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
     "#920000","#924900","#db6d00","#24ff24","#ffff6d"]
c1, c2, c3, c4 =  c[2], c[6], c[10], c[12]

#%%

BG_stft_fs_paths = [] # feature_min_max scaling
BG_wave_fs_paths = [] # feature_min_max scaling
BG_stft_gs_paths = [] # global_min_max scaling
BG_wave_gs_paths = [] # global_min_max scaling
BG_stft_qt_paths = [] # quantile transform scaling
BG_wave_qt_paths = [] # quantile transform scaling
for BG_path in BG_dataset_paths:
    path = os.path.join(BG_parent_path, BG_path)
    rundirs = next(os.walk(path))[1]
    for rundir in rundirs:
        model_path = os.path.join(path, rundir)
        with open(os.path.join(model_path, 'gan_train_config.json'), 'r') as fp:
            train_specs_dict = json.loads(fp.read())
        if train_specs_dict['model_specs']['wavegan'] == True:
            if train_specs_dict['dataloader_specs']['dataset_specs']['quantize'] == 'channel':
                BG_wave_qt_paths.append(model_path)
            else: # no quantile tranformation
                if train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'feature_min_max':
                    BG_wave_fs_paths.append(model_path)
                elif train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'global_min_max':
                    BG_wave_gs_paths.append(model_path)
        else: # stftgan
            if train_specs_dict['dataloader_specs']['dataset_specs']['quantize'] == 'channel':
                BG_stft_qt_paths.append(model_path)
            else: # no quantile tranformation
                if train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'feature_min_max':
                    BG_stft_fs_paths.append(model_path)
                elif train_specs_dict['dataloader_specs']['dataset_specs']['data_scaler'] == 'global_min_max':
                    BG_stft_gs_paths.append(model_path)

#%%


test_set_groups = [BG_wave_fs_paths, BG_stft_fs_paths, BG_wave_gs_paths, BG_stft_gs_paths, BG_wave_qt_paths, BG_stft_qt_paths]
noise_set_names = ["BGN", "BGN", "BGN", "BGN", "BGN", "BGN"]
model_types = ["wavegan", "stftgan", "wavegan_gs", "stftgan_gs", "wavegan_qt", "stftgan_qt"]

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
metrics_df.to_csv(os.path.join(plot_path, "BG_results.csv"), index=False)

print(metrics_df["model_type"].value_counts())

#%%

fig, axs = plt.subplots(nrows=3, ncols=2, sharex="col", figsize = (8,8))
noise_titles = ['Feature Min-Max Data Scaling', 'Quantile Data Transformation']

for i, (suffix, noise_title) in enumerate(zip(["", "_qt"], noise_titles)):
    noise_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_metrics_df = metrics_df[metrics_df["noise_type"] == "BGN"]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"stftgan{suffix}"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"wavegan{suffix}"]

    box_range = list(range(len(stft_metrics_df)))
    target_range = [pos - 0.2 for pos in box_range]
    stft_range = [pos + 0.2 for pos in box_range]

    axs[0, i].plot(box_range, wave_metrics_df["median_psd_dist"], marker="s",
                   color=c2, linestyle="-", alpha=1, label="WaveGAN", linewidth=2)
    axs[0, i].plot(box_range, stft_metrics_df["median_psd_dist"], marker="o",
                   color=c3, linestyle="-", alpha=1, label="STFT-GAN", linewidth=2)
    axs[0, i].set_ylim((0, 0.5))
    axs[0, 0].set_ylabel("Geodesic PSD Distance", fontsize=14)
    axs[0, i].xaxis.set_ticks_position('none')
    axs[0, i].xaxis.set_ticklabels([])
    axs[0, i].grid()
    axs[0, 1].legend(fontsize=14)
    axs[0, i].set_title(noise_title, fontsize=14)


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
    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[1, 0].set_ylabel(r"Estimated $p$", fontsize=14)
    axs[1, i].grid(True)
    axs[1, i].set_ylim(-0.05, 1.0)
    axs[1, i].xaxis.set_ticks_position('none')
    axs[1, i].xaxis.set_ticklabels([])

    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(os.path.join(stft_run["config"], 'parameter_amp_distributions.json'), "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(os.path.join(wave_run["config"], 'parameter_amp_distributions.json'), "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    box1 = axs[2, i].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[2, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[2, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=1), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[2, 1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'],
                  loc='upper center', fontsize=12)

    axs[2, 0].set_ylabel(r"Estimated $\theta$", fontsize=14)
    axs[2, i].set_xlabel(r"Target $p$", fontsize=14)
    axs[2, i].set_xticks(box_range)
    axs[2, i].set_xticklabels(noise_range, fontsize=11, rotation=45)
    axs[2, 0].set_ylim(0, 14)
    axs[2, 1].set_ylim(0, 20)
    axs[0, i].tick_params(axis='y', labelsize=11)
    axs[1, i].tick_params(axis='y', labelsize=11)
    axs[2, i].tick_params(axis='y', labelsize=11)
    axs[2, i].grid(True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.2)
plt.savefig(os.path.join(plot_path, 'BGN_combined_plot.eps'), dpi=300)
plt.show()

#%%
# density and coverage plots

fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', figsize = (8,7))
noise_titles = ['Feature Min-Max Data Scaling', 'Quantile Data Transformation']
parameter_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for i, (suffix, noise_title) in enumerate(zip(["", "_qt"], noise_titles)):
    model_metrics_df = metrics_df[metrics_df["noise_type"] == "BGN"]
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
    if i==0:
        axs[0, 0].errorbar(box_range, wave_metrics_df["dtw_density"], yerr_wave, marker="s",
                             color=c2, linestyle="-", linewidth=2, label="WaveGAN", capsize = 5)
        axs[0, 0].errorbar(box_range, stft_metrics_df["dtw_density"], yerr_stft, marker="o",
                           color=c3, linestyle="-", linewidth=2, label="STFT-GAN", capsize = 5)
        axs[0, 0].set_ylabel("DTW Density", fontsize=14)
        axs[0, 0].set_title(noise_title, fontsize=14)
        axs[0, 0].xaxis.set_ticks_position('none')
        axs[0, 0].xaxis.set_ticklabels([])
        axs[0, 0].tick_params(axis='y', labelsize=12)
        axs[0, 0].grid(True)
        # add legend, removing the errorbars from markers
        handles, labels = axs[0, 0].get_legend_handles_labels()
        handles = [h[0] for h in handles]
        axs[0, 0].legend(handles, labels, loc='upper left', fontsize=14)
    elif i==1:
        # use broken y-axis for density plot
        ax = axs[0, 1]
        divider = make_axes_locatable(ax)
        ax2 = divider.new_vertical(size="100%", pad=0.15)
        fig.add_axes(ax2)
        ax.errorbar(box_range, stft_metrics_df["dtw_density"], yerr_stft, marker="o",
                       color=c3, linestyle="-", linewidth=2, label="STFT-GAN", capsize = 5)
        ax.set_ylim(-0.1, 2)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis='y', labelsize=12)
        ax2.errorbar(box_range, wave_metrics_df["dtw_density"], yerr_wave, marker="s",
                            color=c2, linestyle="-", linewidth=2, label="WaveGAN", capsize = 5)
        ax2.set_ylim(8, 42)
        ax2.set_xticks(box_range)
        ax2.tick_params(bottom=False, labelbottom=False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_title(noise_title, fontsize=14)
        ax2.grid(True, 'both')
        ax2.tick_params(axis='y', labelsize=12)
        ax2.set_yticks([10, 20, 30, 40])

        # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
        # make the diagonal lines for broken axis
        d = .015  # make the diagonal lines for broken axis
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

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
    #axs[1, i].set_ylim((-0.01, 1.02))
    axs[1, i].set_xlabel(r"Target $p$", fontsize=14)
    axs[1, i].set_xticks(box_range)
    axs[1, i].set_xticklabels(parameter_range, fontsize=12, rotation=45)


fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.2)
fig.savefig(os.path.join(plot_path, 'BGN_density_coverage.eps'), dpi=300)
fig.show()


#%%
# plot example target and generated waveforms

path_stft = [x for x in BG_stft_qt_paths if re.search('IP5/', x)]
stftpath = os.path.join(str(path_stft[0]), 'gen_distribution.h5')
path_wave = [x for x in BG_wave_qt_paths if re.search('IP5/', x)]
wavepath = os.path.join(str(path_wave[0]), 'gen_distribution.h5')
targetpath = "../Datasets/BG/BG_fixed_IP5/test.h5"
output_path = os.path.join(plot_path, "BG_waveform_comp_IP5.eps")
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
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.1)
fig.align_ylabels(axs[:,0])
plt.savefig(output_path, dpi=300)
plt.show()

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

# BG
x = np.linspace(-3.5,3.5,1000)
sig_w=0.1
sig_i=1
impulse_prob = [.01, .05, .1]
fig3, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fig4, axs = plt.subplots(len(impulse_prob), 1, sharex=True, sharey=True, figsize=(8, 5))
linestyles = ['-', '--', '-.']
for k, p in enumerate(impulse_prob):
    f_BG = bg.bg_pdf(x, p, sig_w, sig_i)
    waveform = bg.simulate_bg_noise(signal_length, p, sig_w, sig_i)
    ax1.semilogy(x,f_BG,label=f'p={p}', linewidth=3, linestyle=linestyles[k])
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
fig3.savefig(os.path.join(plot_path, 'bg_pdfs.eps'), dpi=300)
fig4.tight_layout(pad=3, h_pad=0.1)
fig4.supxlabel('Time Index', fontsize=axislabelfont1)
fig4.supylabel('Amplitude', fontsize=axislabelfont1)
fig4.savefig(os.path.join(plot_path,'bg_examples.eps'), dpi=300)