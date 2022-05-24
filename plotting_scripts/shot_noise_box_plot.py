import json
import pandas as pd
import matplotlib.pyplot as plt
plot_path = "./paper_plots/"

SNGE_parent_path = "./model_results/shot_noise/SNGE/"
SNOE_parent_path = "./model_results/shot_noise/SNOE/"
SNLE_parent_path = "./model_results/shot_noise/SNLE/"
SNGE_dataset_paths = ["noise_SNGE_shot_gaussian_exponential_fixed_ER25/", "noise_SNGE_shot_gaussian_exponential_fixed_ER50/",
                      "noise_SNGE_shot_gaussian_exponential_fixed_ER75/", "noise_SNGE_shot_gaussian_exponential_fixed_ER100/",
                      "noise_SNGE_shot_gaussian_exponential_fixed_ER125/", "noise_SNGE_shot_gaussian_exponential_fixed_ER150/",
                      "noise_SNGE_shot_gaussian_exponential_fixed_ER175/", "noise_SNGE_shot_gaussian_exponential_fixed_ER200/",
                      "noise_SNGE_shot_gaussian_exponential_fixed_ER225/", "noise_SNGE_shot_gaussian_exponential_fixed_ER250/",
                      "noise_SNGE_shot_gaussian_exponential_fixed_ER275/", "noise_SNGE_shot_gaussian_exponential_fixed_ER300/"]
SNOE_dataset_paths = ["noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER25/", "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER50/",
                      "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER75/", "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER100/",
                      "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER125/", "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER150/",
                      "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER175/", "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER200/",
                      "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER225/", "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER250/",
                      "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER275/", "noise_SNOE_shot_one_sided_exponential_exponential_fixed_ER300/"]
SNLE_dataset_paths = ["noise_SNLE_shot_linear_exponential_exponential_fixed_ER25/", "noise_SNLE_shot_linear_exponential_exponential_fixed_ER50/",
                      "noise_SNLE_shot_linear_exponential_exponential_fixed_ER75/", "noise_SNLE_shot_linear_exponential_exponential_fixed_ER100/",
                      "noise_SNLE_shot_linear_exponential_exponential_fixed_ER125/", "noise_SNLE_shot_linear_exponential_exponential_fixed_ER150/",
                      "noise_SNLE_shot_linear_exponential_exponential_fixed_ER175/", "noise_SNLE_shot_linear_exponential_exponential_fixed_ER200/",
                      "noise_SNLE_shot_linear_exponential_exponential_fixed_ER225/", "noise_SNLE_shot_linear_exponential_exponential_fixed_ER250/",
                      "noise_SNLE_shot_linear_exponential_exponential_fixed_ER275/", "noise_SNLE_shot_linear_exponential_exponential_fixed_ER300/"]

SNGE_wave_paths = [SNGE_parent_path + path + "wavegan/" for path in SNGE_dataset_paths]
SNGE_wave_ps_paths = [SNGE_parent_path + path + "wavegan_ps/" for path in SNGE_dataset_paths]
SNGE_stft_paths = [SNGE_parent_path + path + "stftgan/" for path in SNGE_dataset_paths]
SNOE_wave_paths = [SNOE_parent_path + path + "wavegan/" for path in SNOE_dataset_paths]
SNOE_wave_ps_paths = [SNOE_parent_path + path + "wavegan_ps/" for path in SNOE_dataset_paths]
SNOE_stft_paths = [SNOE_parent_path + path + "stftgan/" for path in SNOE_dataset_paths]
SNLE_wave_paths = [SNLE_parent_path + path + "wavegan/" for path in SNLE_dataset_paths]
SNLE_wave_ps_paths = [SNLE_parent_path + path + "wavegan_ps/" for path in SNLE_dataset_paths]
SNLE_stft_paths = [SNLE_parent_path + path + "stftgan/" for path in SNLE_dataset_paths]

test_set_groups = [SNGE_wave_paths, SNGE_wave_ps_paths, SNGE_stft_paths, SNOE_wave_paths, SNOE_wave_ps_paths,
                   SNOE_stft_paths, SNLE_wave_paths, SNLE_wave_ps_paths, SNLE_stft_paths]
noise_set_names = ["SNGE", "SNGE", "SNGE", "SNOE", "SNOE", "SNOE", "SNLE", "SNLE", "SNLE"]
model_types = ["wavegan", "wavegan_ps", "stftgan", "wavegan", "wavegan_ps", "stftgan", "wavegan", "wavegan_ps", "stftgan"]
df_temp = []
for test_set, noise_set, model in zip(test_set_groups, noise_set_names, model_types):
    for model_path in test_set:
        print(model_path)
        with open(model_path + "distance_metrics.json") as f:
            data = json.load(f)
            data["config"] = model_path
            data["model_type"] = model
            data["noise_type"] = noise_set
            with open(model_path + "gan_train_config.json") as f2:
                train_dict = json.load(f2)
                data['dataset'] = train_dict['dataloader_specs']['dataset_specs']['data_set']
            df_temp.append(data)
metrics_df = pd.DataFrame(df_temp)
metrics_df.to_csv(plot_path + "shotnoise_results.csv", index=False)


#%%

noise_types = ["SNOE", "SNLE", "SNGE"]
noise_titles = ["One-sided Exponential Pulse Type", "Linear Exponential Pulse Type", "Gaussian Pulse Type"]
noise_range = [0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]

fig, axs = plt.subplots(nrows=2, ncols=3, sharey='row') #, sharex="col")
fig.set_figheight(7)
fig.set_figwidth(12)
for i, (noise_type, noise_title) in enumerate(zip(noise_types, noise_titles)):
    model_metrics_df = metrics_df[metrics_df["noise_type"] == noise_type]
    stft_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "stftgan"]
    wave_metrics_df = model_metrics_df[model_metrics_df["model_type"] == "wavegan_ps"]
    box_range = list(range(len(stft_metrics_df)))
    target_range = [pos - 0.2 for pos in box_range]
    wave_range = box_range
    stft_range = [pos + 0.2 for pos in box_range]
    c1, c2, c3 = "red", "blue", "green"

    axs[0, i].plot(box_range, stft_metrics_df["geodesic_psd_dist"], marker="o", color="blue", linestyle="-", alpha=0.7, label="STFT-GAN")
    axs[0, i].plot(box_range, wave_metrics_df["geodesic_psd_dist"], marker="o", color="red", linestyle="-", alpha=0.7, label="WaveGAN")
    axs[0, i].set_ylim((0, 2.3))
    axs[0, i].set_title(noise_title, fontsize=12)
    axs[0, i].xaxis.set_ticks_position('none')
    axs[0, i].xaxis.set_ticklabels([])
    axs[0, 0].set_ylabel(r"$d_g$", fontsize=12)
    axs[1, i].set_xlabel(r"True $\nu$", fontsize=12)
    axs[0, i].grid()
    axs[0, 0].legend(fontsize=12, loc="upper left")

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
    box1 = axs[1, i].boxplot(wave_dists, showfliers=False, positions=wave_range, widths=0.2, notch=True, patch_artist=True,
                       boxprops=dict(facecolor=c1, color=c1, alpha=0.7), medianprops=dict(color='black'),
                       capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box2 = axs[1, i].boxplot(stft_dists, showfliers=False, positions=stft_range, widths=0.2, notch=True, patch_artist=True,
                       boxprops=dict(facecolor=c2, color=c2, alpha=0.7), medianprops=dict(color='black'),
                       capprops=dict(color="black"), whiskerprops=dict(color="black"))
    box3 = axs[1, i].boxplot(target_dists, showfliers=False, positions=target_range, widths=0.2, notch=True, patch_artist=True,
                       boxprops=dict(facecolor=c3, color=c3, alpha=0.7), medianprops=dict(color='black'),
                       capprops=dict(color="black"), whiskerprops=dict(color="black"))

    axs[1, 0].legend([box3["boxes"][0], box1["boxes"][0], box2["boxes"][0]], ['Target', 'WaveGAN', 'STFT-GAN'], loc='upper left', fontsize=12)
    axs[1, 0].set_ylabel(r"Estimated $\nu$", fontsize=12)
    axs[1, i].grid(True)
    axs[1, i].set_xticks(target_range)
    axs[1, i].set_xticklabels(noise_range, fontsize=10, rotation=45)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f'{plot_path}shotnoise_combined_plots.png', dpi=600)
plt.show()

