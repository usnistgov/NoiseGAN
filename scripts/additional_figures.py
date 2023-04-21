# -*- coding: utf-8 -*-
"""
script for additional figures
"""

import os
import h5py
import os.path
import numpy as np
from spectrum import pmtm
import matplotlib.pyplot as plt
#plt.style.use('seaborn-colorblind')
c = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
     "#920000","#924900","#db6d00","#24ff24","#ffff6d"]
c1, c2, c3, c4 =  c[2], c[6], c[10], c[12]

parent_dataset_path = "../Datasets/"
plot_path = "../paper_plots/"
if not os.path.exists(plot_path):
   os.makedirs(plot_path)

#%%

def get_median_targ_psd(path):
    print(path)
    if os.path.exists(os.path.join(parent_dataset_path, f"{path}/median_psd.h5")):
        h5f = h5py.File(os.path.join(parent_dataset_path, f"{path}/median_psd.h5"), 'r')
        targ_median_psd = h5f['median_psd'][:]
        h5f.close()
    else:
        h5f = h5py.File(os.path.join(parent_dataset_path, f"{path}/test.h5"), 'r')
        targ_dataset = h5f['train'][:]
        h5f.close()
        targ_data = np.array(targ_dataset[:, 1:]).astype(float)
        targ_PSD_estimates = []
        for i, targ_sample in enumerate(targ_data):
            if i % 100 == 0:
                print(f"estimating PSD {i} of {len(targ_data)}")
            Sk, weights, _eigenvalues = pmtm(targ_sample, NW=4, k=7, method='eigen')
            Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
            Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
            targ_PSD_estimates.append(Pxx)
        targ_median_psd = np.median(np.array(targ_PSD_estimates), axis=0)
        h5f = h5py.File(os.path.join(parent_dataset_path, f"{path}/median_psd.h5"), 'w')
        h5f.create_dataset('median_psd', data=targ_median_psd)
        h5f.close()
    return targ_median_psd

#%%
# plot example target PSDs for shot noise

snot_noise_1 = ["shot/shot_one_sided_exponential_exponential_fixed_ER100", "shot/shot_gaussian_exponential_fixed_ER100"]
shot_noise_lables = ["One-sided Exponential", "Gaussian"]

colors = [c1, c3]
linestyles = ['-', '--']
fig = plt.figure(figsize=(8,5))
for path, label, color, linestyle in zip(snot_noise_1, shot_noise_lables, colors, linestyles):
    targ_median_psd = get_median_targ_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    plt.plot(w, 10 * np.log10(targ_median_psd), alpha=1, color=color, linestyle=linestyle, linewidth=3, label=label)
plt.ylabel('Power Density (dB)', fontsize=16)
plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=16)
plt.grid()
plt.legend(fontsize=14)
plt.ylim((-80, 20))
plt.xlim((-0.005, 0.5))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(plot_path, "shot_noise_target_psd_comparison_ER100.png"), dpi=600)
plt.show()

#%%
"""
# plot example target PSDs for power law noise

frac_noise_1 = ["FBM/FBM_fixed_H20/", "FGN/FGN_fixed_H20/"]
frac_noise_label = ["FBM", "FGN"]
frac_noise_2 = ["FBM/FBM_fixed_H80/", "FGN/FGN_fixed_H80/"]

fig, axs = plt.subplots(2, figsize=(8, 8), sharex=True, gridspec_kw={'hspace': 0.1})
for path, label, color, linestyle in zip(frac_noise_1, frac_noise_label, colors, linestyles):
    targ_median_psd = get_median_targ_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    axs[0].plot(w, 10 * np.log10(targ_median_psd), alpha=0.85, color=color, linestyle=linestyle, label=label, linewidth=3)
axs[0].set_ylabel('Power Density (dB)', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[0].grid()
axs[0].margins(x=0)
axs[0].legend(fontsize=14)
plt.xscale("log")

for path, label, color, linestyle in zip(frac_noise_2, frac_noise_label, colors, linestyles):
    targ_median_psd = get_median_targ_psd(path)
    w = np.linspace(0, 0.5, len(targ_median_psd))
    axs[1].plot(w, 10 * np.log10(targ_median_psd), alpha=0.85, color=color, linestyle=linestyle, label=label, linewidth=3)
axs[1].set_ylabel('Power Density (dB)', fontsize=16)
plt.xlabel("Normalized Digital Frequency (cycles/sample)", fontsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=14)
axs[1].grid()
axs[1].margins(x=0)
plt.xscale("log")
plt.savefig(os.path.join(plot_path, "frac_noise_target_psd_comparison.png"), dpi=600)
plt.show()

"""