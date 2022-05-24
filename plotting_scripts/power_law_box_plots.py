import json
import pandas as pd
import matplotlib.pyplot as plt

plot_path = "./paper_plots/"

FDWN_parent_path = "./model_results/power_law_noise/FDWN/"
FGN_parent_path = "./model_results/power_law_noise/FGN/"
FBM_parent_path = "./model_results/power_law_noise/FBM/"
FDWN_dataset_paths = ["FDWN_fixed_H5/", "FDWN_fixed_H10/", "FDWN_fixed_H20/", "FDWN_fixed_H30/", "FDWN_fixed_H40/",
                      "FDWN_fixed_H50/", "FDWN_fixed_H60/", "FDWN_fixed_H70/", "FDWN_fixed_H80/", "FDWN_fixed_H90/", "FDWN_fixed_H95/"]
FGN_dataset_paths = ["FGN_fixed_H5/", "FGN_fixed_H10/", "FGN_fixed_H20/", "FGN_fixed_H30/", "FGN_fixed_H40/", "FGN_fixed_H50/",
                     "FGN_fixed_H60/", "FGN_fixed_H70/", "FGN_fixed_H80/", "FGN_fixed_H90/", "FGN_fixed_H95/"]
FBM_dataset_paths = ["FBM_fixed_H5/", "FBM_fixed_H10/", "FBM_fixed_H20/", "FBM_fixed_H30/", "FBM_fixed_H40/",
                     "FBM_fixed_H50/", "FBM_fixed_H60/", "FBM_fixed_H70/", "FBM_fixed_H80/", "FBM_fixed_H90/", "FBM_fixed_H95/"]
FDWN_stft_paths = [FDWN_parent_path + path + "stftgan/" for path in FDWN_dataset_paths]
FDWN_wave_paths = [FDWN_parent_path + path + "wavegan/" for path in FDWN_dataset_paths]
FDWN_wave_ps_paths = [FDWN_parent_path + path + "wavegan_ps/" for path in FDWN_dataset_paths]

FGN_stft_paths = [FGN_parent_path + path + "stftgan/" for path in FGN_dataset_paths]
FGN_wave_paths = [FGN_parent_path + path + "wavegan/" for path in FGN_dataset_paths]
FGN_wave_ps_paths = [FGN_parent_path + path + "wavegan_ps/" for path in FGN_dataset_paths]

FBM_stft_paths = [FBM_parent_path + path + "stftgan/" for path in FBM_dataset_paths]
FBM_wave_paths = [FBM_parent_path + path + "wavegan/" for path in FBM_dataset_paths]
FBM_wave_ps_paths = [FBM_parent_path + path + "wavegan_ps/" for path in FBM_dataset_paths]
FBM_stft2_paths = [FBM_parent_path + path + "stftgan2/" for path in FBM_dataset_paths]
test_set_groups = [FBM_wave_paths, FBM_stft_paths, FBM_stft2_paths, FBM_wave_ps_paths, FGN_wave_paths,
                   FGN_stft_paths, FGN_wave_ps_paths, FDWN_wave_paths, FDWN_stft_paths, FDWN_wave_ps_paths]
noise_set_names = ["FBM", "FBM", "FBM", "FBM", "FGN", "FGN", "FGN", "FDWN", "FDWN", "FDWN"]
model_types = ["wavegan", "stftgan", "stftgan2", "wavegan_ps", "wavegan",
               "stftgan", "wavegan_ps", "wavegan", "stftgan", "wavegan_ps"]
df_temp = []
for test_set, noise_set, model in zip(test_set_groups, noise_set_names, model_types):
    for model_path in test_set:
        with open(model_path + "distance_metrics.json") as f:
            data = json.load(f)
            data["config"] = model_path
            data["dataset"] = "/".join(data["config"].split("/")[3:5])
            data["model_type"] = model
            print(f"{data['dataset']}, {model}, {noise_set}")
            data["noise_type"] = noise_set
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(plot_path + "powerlaw_results.csv", index=False)


#%%
noise_types = ["FDWN", "FGN","FBM"]
x_labels = [r"$d$", r"$H$", r"$H$"]
noise_titles = ["Fractionally Differenced White Noise","Fractional Gaussian Noise", "Fractional Brownian Motion"]
hurst_indices = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
d_frac = [-0.45, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.45]
noise_param_ranges = {"FBM": hurst_indices, "FGN": hurst_indices, "FDWN": d_frac}
fig, axs = plt.subplots(nrows=2, ncols=3, sharex="col")
fig.set_figheight(7)
fig.set_figwidth(14)

for i, (noise_type, x_label, noise_title) in enumerate(zip(noise_types, x_labels, noise_titles)):
    noise_range = noise_param_ranges[noise_type]
    model_metrics_df = metrics_df[metrics_df["noise_type"] == noise_type]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan_ps"]
    stft2_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan2"] if noise_type == "FBM" else None
    stft_dists, stft2_dists, target_dists, wave_dists = [], [], [], []
    for _, stft_run in stft_metrics_df.iterrows():
        f = open(f'{stft_run["config"]}parameter_value_distributions.json', "r")
        stft_param_dists = json.loads(f.read())
        target_dists.append(stft_param_dists["target"])
        stft_dists.append(stft_param_dists["generated"])
    for _, wave_run in wave_metrics_df.iterrows():
        f = open(f'{wave_run["config"]}parameter_value_distributions.json', "r")
        wave_param_dists = json.loads(f.read())
        wave_dists.append(wave_param_dists["generated"])
    if noise_type == "FBM":
        for _, stft2_run in stft2_metrics_df.iterrows():
            f = open(f'{stft2_run["config"]}parameter_value_distributions.json', "r")
            stft_param_dists = json.loads(f.read())
            stft2_dists.append(stft_param_dists["generated"])

    box_range = list(range(len(stft_dists)))
    if noise_type == "FBM":
        target_range = [pos - 0.3 for pos in box_range]
        wave_range = [pos - 0.1 for pos in box_range]
        stft_range = [pos + 0.1 for pos in box_range]
        stft2_range = [pos + 0.3 for pos in box_range]
    else:
        target_range = [pos - 0.2 for pos in box_range]
        wave_range = box_range
        stft_range = [pos + 0.2 for pos in box_range]

    c1, c2, c3, c4 = "red", "blue", "green", "purple"
    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=wave_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c1, color=c1, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c2, color=c2, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                          boxprops=dict(facecolor=c3, color=c3, alpha=0.7), medianprops=dict(color='black'),
                          capprops=dict(color="black"), whiskerprops=dict(color="black"))
    if stft2_metrics_df is not None:
        box4 = axs[1, i].boxplot(stft2_dists, showfliers=False, positions=stft2_range, widths=0.2, notch=True, patch_artist=True,
                              boxprops=dict(facecolor=c4, color=c4, alpha=0.7), medianprops=dict(color='black'),
                              capprops=dict(color="black"), whiskerprops=dict(color="black"))
    if noise_type == "FBM":
        axs[1, 2].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0], box4["boxes"][0]],
                   ['Target', 'WaveGAN', 'STFT-GAN (65x65)', 'STFT-GAN (129x65)'], loc='upper left', fontsize=12)
    axs[1, i].set_ylabel(fr"Estimated {x_label}", fontsize=12)
    axs[1, i].grid(True)
    axs[1, i].xaxis.set_ticks_position('none')
    axs[1, i].xaxis.set_ticklabels([])

    axs[0, i].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="o", color="red", linestyle="-", alpha=0.7,
             label="WaveGAN")
    axs[0, i].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o", color="blue", linestyle="-", alpha=0.7,
             label="STFT-GAN" if noise_type != "FBM" else 'STFT-GAN (65x65)')
    if noise_type == "FBM":
        axs[0, i].plot(box_range, stft2_metrics_df["geodesic_psd_dist"], marker="o", color="purple", linestyle="-", alpha=0.7,
                 label="STFT-GAN (129x65)")
    axs[0, i].set_ylim((0, 1.3))
    axs[0, i].set_title(noise_title, fontsize=12)
    axs[0, i].set_xticks(box_range)
    axs[0, i].set_xticklabels(noise_range, fontsize=10)
    axs[0, i].set_ylabel("$d_g$", fontsize=12)
    axs[1, i].set_xlabel(fr"True {x_label}", fontsize=12)
    axs[0, i].grid()
    axs[0, 2].legend(fontsize=12)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.15)
plt.savefig(f'{plot_path}power_law_combined_plot.png', dpi=600)
plt.show()
