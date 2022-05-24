import json
import pandas as pd
import matplotlib.pyplot as plt


plot_path = "./paper_plots/"
BGN_parent_path = "./model_results/impulsive_noise/BGN/"
BGN_parent_path_2 = "./model_results/impulsive_noise/BGN_tests/"
SAS_parent_path = "./model_results/impulsive_noise/SAS/"
BGN_quant_parent_path = "./model_results/impulsive_noise/BGN_quant/"
SAS_quant_parent_path = "./model_results/impulsive_noise/SAS_quant/"
BGN_dataset_paths = ["BGN_BG_fixed_IP1/", "BGN_BG_fixed_IP5/", "BGN_BG_fixed_IP10/", "BGN_BG_fixed_IP15/",
                     "BGN_BG_fixed_IP20/", "BGN_BG_fixed_IP30/", "BGN_BG_fixed_IP40/", "BGN_BG_fixed_IP50/",
                     "BGN_BG_fixed_IP60/", "BGN_BG_fixed_IP70/", "BGN_BG_fixed_IP80/", "BGN_BG_fixed_IP90/"]
SAS_dataset_paths = ["SAS_SAS_fixed_alpha50/", "SAS_SAS_fixed_alpha60/", "SAS_SAS_fixed_alpha70/", "SAS_SAS_fixed_alpha80/",
                     "SAS_SAS_fixed_alpha90/", "SAS_SAS_fixed_alpha100/", "SAS_SAS_fixed_alpha110/", "SAS_SAS_fixed_alpha120/",
                     "SAS_SAS_fixed_alpha130/", "SAS_SAS_fixed_alpha140/", "SAS_SAS_fixed_alpha150/"]
BGN_wave_ps_paths = [BGN_parent_path + path + "wavegan_ps/" for path in BGN_dataset_paths]
BGN_wave_paths = [BGN_parent_path + path + "wavegan/" for path in BGN_dataset_paths]
BGN_stft_paths = [BGN_parent_path + path + "stftgan/" for path in BGN_dataset_paths]
BGN_stft2_paths = [BGN_parent_path_2 + path + "stftgan/" for path in BGN_dataset_paths]
SAS_wave_ps_paths = [SAS_parent_path + path + "wavegan_ps/" for path in SAS_dataset_paths]
SAS_wave_paths = [SAS_parent_path + path + "wavegan/" for path in SAS_dataset_paths]
SAS_stft_paths = [SAS_parent_path + path + "stftgan/" for path in SAS_dataset_paths]
BGN_quantized_wave_ps_paths = [BGN_quant_parent_path + path + "wavegan_ps/" for path in BGN_dataset_paths]
BGN_quantized_wave_paths = [BGN_quant_parent_path + path + "wavegan/" for path in BGN_dataset_paths]
BGN_quantized_stft_paths = [BGN_quant_parent_path + path + "stftgan/" for path in BGN_dataset_paths]
SAS_quantized_wave_ps_paths = [SAS_quant_parent_path + path + "wavegan_ps/" for path in SAS_dataset_paths]
SAS_quantized_wave_paths = [SAS_quant_parent_path + path + "wavegan/" for path in SAS_dataset_paths]
SAS_quantized_stft_paths = [SAS_quant_parent_path + path + "stftgan/" for path in SAS_dataset_paths]
test_set_groups = [BGN_wave_paths, BGN_stft_paths, BGN_stft2_paths, BGN_quantized_wave_paths, BGN_quantized_stft_paths,
                   BGN_wave_ps_paths, BGN_quantized_wave_ps_paths, SAS_wave_paths, SAS_stft_paths, SAS_quantized_wave_paths,
                   SAS_quantized_stft_paths, SAS_wave_ps_paths, SAS_quantized_wave_ps_paths]
noise_set_names = ["BGN", "BGN", "BGN", "BGN", "BGN", "BGN", "BGN", "SAS", "SAS", "SAS", "SAS", "SAS", "SAS"]
model_types = ["wavegan", "stftgan", "stftgan2", "wavegan_quantized", "stftgan_quantized", "wavegan_ps", "wavegan_ps_quantized",
               "wavegan", "stftgan", "wavegan_quantized", "stftgan_quantized", "wavegan_ps", "wavegan_ps_quantized"]

df_temp = []
for test_set, noise_set, model in zip(test_set_groups, noise_set_names, model_types):
    for model_path in test_set:
        with open(model_path + "distance_metrics.json") as f:
            data = json.load(f)
            data["config"] = model_path
            data["model_type"] = model
            data["noise_type"] = noise_set
            with open(model_path + "gan_train_config.json") as f2:
                train_dict = json.load(f2)
                data['dataset'] = train_dict['dataloader_specs']['dataset_specs']['data_set']
            print(f"{data['dataset']}, {model}, {noise_set}")
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(plot_path + "impulsive_results.csv", index=False)

print(metrics_df["model_type"].value_counts())

#%%

fig, axs = plt.subplots(nrows=3, ncols=2, sharex="col")
fig.set_figheight(8)
fig.set_figwidth(8)

for i, suffix in enumerate(["", "_quantized"]):
    noise_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_metrics_df = metrics_df[metrics_df["noise_type"] == "BGN"]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"stftgan{suffix}"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"wavegan_ps{suffix}"]

    box_range = list(range(len(stft_metrics_df)))
    target_range = [pos - 0.2 for pos in box_range]
    stft_range = [pos + 0.2 for pos in box_range]
    c1, c2, c3 = "red", "blue", "green"

    axs[0, i].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="o", color="red", linestyle="-", alpha=0.7, label="WaveGAN")
    axs[0, i].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o", color="blue", linestyle="-", alpha=0.7, label="STFT-GAN")
    axs[0, i].set_ylim((0, 0.4))
    axs[0, 0].set_ylabel(r"$d_g$", fontsize=12)
    axs[0, i].xaxis.set_ticks_position('none')
    axs[0, i].xaxis.set_ticklabels([])
    axs[0, i].grid()

    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(f'{stft_run["config"]}parameter_value_distributions.json', "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(f'{wave_run["config"]}parameter_value_distributions.json', "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[1, 0].set_ylabel(r"Estimated $p$", fontsize=12)
    axs[1, i].grid(True)
    axs[1, i].xaxis.set_ticks_position('none')
    axs[1, i].xaxis.set_ticklabels([])

    stft_dists, target_dists, wave_dists = [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(f'{stft_run["config"]}parameter_amp_distributions.json', "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(f'{wave_run["config"]}parameter_amp_distributions.json', "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    box1 = axs[2, i].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[2, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[2, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    axs[2, 1].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'],
                  loc='upper right', fontsize=12)
    axs[2, 0].set_ylabel(r"Estimated $\theta$", fontsize=12)
    axs[2, i].set_xlabel(r"True $p$", fontsize=12)
    axs[2, i].set_xticks(box_range, )
    axs[2, i].set_xticklabels(noise_range, fontsize=10, rotation=45)
    axs[0, i].tick_params(axis='y', labelsize=10)
    axs[1, i].tick_params(axis='y', labelsize=10)
    axs[2, i].tick_params(axis='y', labelsize=10)
    axs[2, i].grid(True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.15)
plt.savefig(f'{plot_path}BGN_combined_plot.png', dpi=600)
plt.show()


#%%
import numpy as np


fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
fig.set_figheight(7)
fig.set_figwidth(5)
noise_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
model_metrics_df = metrics_df[metrics_df["noise_type"] == "SAS"]
stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"stftgan_quantized"]
wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == f"wavegan_ps_quantized"]
box_range = list(range(len(stft_metrics_df)))
target_range = [pos - 0.2 for pos in box_range]
stft_range = [pos + 0.2 for pos in box_range]
c1, c2, c3 = "red", "blue", "green"

axs[0].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="o", color="red", linestyle="-", alpha=0.7, label="WaveGAN")
axs[0].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o", color="blue", linestyle="-", alpha=0.7, label="STFT-GAN")
axs[0].set_ylim((0, 0.8))
axs[0].set_ylabel(r"$d_g$", fontsize=12)
axs[0].grid(True)
axs[0].xaxis.set_ticks_position('none')
axs[0].xaxis.set_ticklabels([])

stft_dists, target_dists, wave_dists = [], [], []
for _, stft_run in stft_metrics_df.iterrows():
    f = open(f'{stft_run["config"]}parameter_value_distributions.json', "r")
    stft_param_dists = json.loads(f.read())
    target_dists.append(stft_param_dists["target"])
    stft_dists.append(stft_param_dists["generated"])
for _, wave_run in wave_metrics_df.iterrows():
    f = open(f'{wave_run["config"]}parameter_value_distributions.json', "r")
    wave_param_dists = json.loads(f.read())
    wave_dists.append(wave_param_dists["generated"])
stft_dists = np.array(stft_dists)
stft_dists = [row[~np.isnan(row)] for row in stft_dists]
box1 = axs[1].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                      boxprops=dict(facecolor=c1, color=c1, alpha=0.7), medianprops=dict(color='black'),
                      capprops=dict(color="black"), whiskerprops=dict(color="black"))
box2 = axs[1].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                      boxprops=dict(facecolor=c2, color=c2, alpha=0.7), medianprops=dict(color='black'),
                      capprops=dict(color="black"), whiskerprops=dict(color="black"))
box3 = axs[1].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                      boxprops=dict(facecolor=c3, color=c3, alpha=0.7), medianprops=dict(color='black'),
                      capprops=dict(color="black"), whiskerprops=dict(color="black"))
axs[1].set_ylabel(r"Estimated $\alpha$", fontsize=12)
axs[1].grid(True)
axs[1].xaxis.set_ticks_position('none')
axs[1].xaxis.set_ticklabels([])

stft_dists, target_dists, wave_dists = [], [], []
for _, stft_run in stft_metrics_df.iterrows():
    f = open(f'{stft_run["config"]}parameter_scale_distributions.json', "r")
    stft_param_dists = json.loads(f.read())
    target_dists.append(stft_param_dists["target"])
    stft_dists.append(stft_param_dists["generated"])
for _, wave_run in wave_metrics_df.iterrows():
    f = open(f'{wave_run["config"]}parameter_scale_distributions.json', "r")
    wave_param_dists = json.loads(f.read())
    wave_dists.append(wave_param_dists["generated"])
stft_dists = np.array(stft_dists)
stft_dists = [row[~np.isnan(row)] for row in stft_dists]
box1 = axs[2].boxplot(wave_dists, showfliers=False, positions=box_range, widths=0.2, notch=True, patch_artist=True,
                         boxprops=dict(facecolor=c1, color=c1, alpha=0.7), medianprops=dict(color='black'),
                         capprops=dict(color="black"), whiskerprops=dict(color="black"))
box2 = axs[2].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                         boxprops=dict(facecolor=c2, color=c2, alpha=0.7), medianprops=dict(color='black'),
                         capprops=dict(color="black"), whiskerprops=dict(color="black"))
box3 = axs[2].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                         boxprops=dict(facecolor=c3, color=c3, alpha=0.7), medianprops=dict(color='black'),
                         capprops=dict(color="black"), whiskerprops=dict(color="black"))
axs[2].set_ylabel(r"Estimated $\gamma$", fontsize=12)
axs[2].grid(True)
axs[2].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'], loc='upper right', fontsize=12)
axs[2].set_xticks(box_range)
axs[2].set_xticklabels(noise_range, fontsize=10, rotation=45)
axs[2].set_xlabel(r"True $\alpha$", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f'{plot_path}SAS_combined_plot_2.png', dpi=600)
plt.show()
