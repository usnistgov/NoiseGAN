#!/usr/bin/env python3
"""
Module for evaluating generated noise datasets.

"""

import os
import json
import h5py
import scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spectrum import pmtm
import utils.shot_noise_utils as sn
import utils.fractional_noise_utils as fn
from utils.alpha_stable_noise_utils import estimate_alpha_fast, estimate_scale_fast
from utils.bg_noise_utils import estimate_bg_parameters


def evaluate_bp_noise(gen_data, gen_labels, targ_data, targ_labels, save_path):
    """
    Evaluate bandpass (BP) noise dataset.

    Parameters
    ----------
    gen_data : ndarray
        Generated data.
    gen_labels : array-like, Iterable, dict, or scalar value.
        Labels for generated data.
    targ_data : ndarray
        Target data.
    targ_labels : array-like, Iterable, dict, or scalar value.
        Labels for target data.
    save_path : string
        Directory to save results.

    Returns
    -------
    None.

    """
    NW = 4  # time-halfbandwidth product for pmtm psd estimator
    _ = plt.figure(figsize=(10, 7))
    gen_labels, targ_labels = pd.Series(gen_labels), pd.Series(targ_labels)
    gen_data, targ_data = pd.DataFrame(gen_data), pd.DataFrame(targ_data)
    gen_dataset = pd.concat([gen_labels, gen_data], axis=1, ignore_index=True)
    targ_dataset = pd.concat([targ_labels, targ_data], axis=1, ignore_index=True)
    for dataset, label, color in zip([gen_dataset, targ_dataset],
                                     ["Generated_ Distribution", "Target Distribution"], ["blue", "red"]):
        for name, group in dataset.groupby(dataset.columns[0]):
            waveforms = group.iloc[:, 1:].values
            psd_list = []
            for num, waveform in enumerate(waveforms):
                Sk, weights, _eigenvalues = pmtm(waveform, NW, k=2 * NW - 1, method='eigen')
                Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
                Pxx = Pxx[0: (len(Pxx) // 2)]
                psd_list.append(Pxx)
            median_psd = np.median(np.array(psd_list), axis=0)
            median_psd = median_psd / np.mean(median_psd[np.where(median_psd > 1e-4)])
            w = np.linspace(0, 1, len(median_psd))
            plt.plot(w, 10 * np.log10(median_psd), color=color)
    plt.xlabel('Normalized Digital Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'PSD_plot.png'), dpi=300)
        plt.close('all')
    plt.clf()


def evaluate_fn(targ_data, gen_data, noise_type, output_path):
    """
    Evaluate bandpass fractional (power law) noise dataset.

    Parameters
    ----------
    targ_data : ndarray
        Target data.
    gen_data : ndarray
        Generated data.
    noise_type : string
        Fractional noise type. Options: 'FGN', 'FBM', 'FDWN'
    output_path : string, optional
        Directory to save results.

    Returns
    -------
    targ_H_estimates : list
        DESCRIPTION.
    gen_H_estimates : list
        DESCRIPTION.
    hurst_wass_dist : float
        Wasserstein distance between distributions of estimated Hurst indices
        for target and generated data distributions.

    """
    targ_H_estimates, gen_H_estimates = [], []
    for i in range(len(targ_data)):
        targ_sample, gen_sample = targ_data[i, :], gen_data[i, :]
        if noise_type == 'FBM' or noise_type == 'FGN':
            targ_H_est = fn.estimate_Hurst_exponent(targ_sample, noise_type)
            gen_H_est = fn.estimate_Hurst_exponent(gen_sample, noise_type)
            targ_H_estimates.append(targ_H_est)
            gen_H_estimates.append(gen_H_est)
        else:
            targ_sample, gen_sample = targ_sample.tolist(), gen_sample.tolist()
            targ_dfrac, _ = fn.estimate_FD_param(targ_sample)
            gen_dfrac, _ = fn.estimate_FD_param(gen_sample)
            targ_H_estimates.append(targ_dfrac)
            gen_H_estimates.append(gen_dfrac)
    param_dists = {"target": targ_H_estimates, "generated": gen_H_estimates}
    with open(os.path.join(output_path, 'parameter_value_distributions.json'), 'w') as f:
        json.dump(param_dists, f)


    bins = np.histogram(np.concatenate((targ_H_estimates, gen_H_estimates)), bins=50)[1]
    plt.hist(targ_H_estimates, bins=bins, color="red", alpha=0.7, label="Target Hurst Indices")
    plt.hist(gen_H_estimates, bins=bins, color="blue", alpha=0.7, label="Generated Hurst Indices")
    plt.grid()
    plt.xlabel("Estimated Hurst Indices")
    plt.ylabel("Count")
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "hurst_histogram.png"), dpi=300)
    plt.clf()

    param_dists = {"target": targ_H_estimates, "generated": gen_H_estimates}
    with open(os.path.join(output_path, 'parameter_value_distributions.json'), 'w') as f:
        json.dump(param_dists, f)

    # Get wasserstein istance between target and generated Hurst index distributions
    hurst_wass_dist = scipy.stats.wasserstein_distance(targ_H_estimates, gen_H_estimates)
    return targ_H_estimates, gen_H_estimates, hurst_wass_dist


def evaluate_sn(targ_data, gen_data, pulse_type, amp_distrib, param_value, output_path):
    """
    Evaluate shot noise dataset.

    Parameters
    ----------
    targ_data : ndarray
        Target data.
    gen_data : ndarray
        Generated data.
    pulse_type : string
        String specifiying pulse shape.
        Options: 'one_sided_exponential', 'linear_exponential', 'gaussian'
    amp_distrib : string
        String specifying pulse amplitude distribution.
        Options: 'exponential', 'rayleigh', 'standard_normal'
    param_value : float
        Target shot noise event rate.
    output_path : string
        Directory to save results.

    Returns
    -------
    None.

    """
    tau_d, beta, theta = 1, 1, 0.1
    gen_nu_ests, targ_nu_ests = [], []
    targ_Rxxs, gen_Rxxs = [], []
    for gen_sample, targ_sample in zip(gen_data, targ_data):
        gen_nu_est = sn.estimate_event_rate(gen_sample, pulse_type, amp_distrib, theta, tau_d)
        targ_nu_est = sn.estimate_event_rate(targ_sample, pulse_type, amp_distrib, theta, tau_d)
        if param_value is None:
            targ_nu_value = targ_nu_est
            gen_nu_value = gen_nu_est
        else:
            targ_nu_value = param_value
            gen_nu_value = param_value

        targ_Rxx, tau, Rxx_theory = sn.estimate_acf(targ_sample, pulse_type, targ_nu_value,
                                                    theta, tau_d, beta, amp_distrib)
        gen_Rxx, _, _ = sn.estimate_acf(gen_sample, pulse_type, gen_nu_value, theta, tau_d, beta,
                                        amp_distrib)
        gen_nu_ests.append(gen_nu_est)
        targ_nu_ests.append(targ_nu_est)
        targ_Rxxs.append(targ_Rxx)
        gen_Rxxs.append(gen_Rxx)

    gen_median_nu_est = np.median(gen_nu_ests)
    targ_median_nu_est = np.median(targ_nu_ests)
    targ_median_acf = np.median(np.array(targ_Rxxs), axis=0)
    gen_median_acf = np.median(np.array(gen_Rxxs), axis=0)

    param_dists = {"target": targ_nu_ests, "generated": gen_nu_ests}
    with open(os.path.join(output_path, 'parameter_value_distributions.json'), 'w') as f:
        json.dump(param_dists, f)

    # get event rate earth movers distance:
    event_rate_dist = scipy.stats.wasserstein_distance(targ_nu_ests, gen_nu_ests)

    bins = np.histogram(np.hstack((gen_nu_ests, targ_nu_ests)), bins=50)[1]  # get the bin edges
    plt.hist(gen_nu_ests, bins=bins, color="blue", alpha=0.7, label="Generated Distribution")
    plt.hist(targ_nu_ests, bins=bins, color="green", alpha=0.7, label="Target Distribution")
    if param_value is not None:
        plt.axvline(x=param_value, color="black", alpha=0.8, label="True Event Rate")
    plt.legend()
    plt.grid()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "event_rate_hists.png"), dpi=300)
    plt.clf()

    plt.plot(tau, targ_median_acf, color='blue', alpha=0.7, label='Target Median ACF')
    plt.plot(tau, gen_median_acf, color='green', alpha=0.7, label='Generated Median ACF')
    plt.title('ACF comparison')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "acf_comparison.png"), dpi=300)
    plt.clf()
    return gen_median_nu_est, targ_median_nu_est, event_rate_dist


def evaluate_bgn(targ_data, gen_data, param_value, output_path):
    """
    Evaluate Bernoulli-Gaussin (BG) noise dataset.

    Parameters
    ----------
    targ_data : ndarray
        Target data.
    gen_data : ndarray
        Generated data.
    param_value : float
        Target impulse probability.
    output_path : string
        Directory to save results.

    Returns
    -------
    targ_prob_median : array
        Median estimated impulse probabilities for target data.
    gen_prob_median : array
        Median estimated impulse probabilities for generated data.
    targ_amp_ratio_median : array
        Median estimated scale ratio for target data.
    gen_amp_ratio_median : array
        Median estimation scale ratio for generated data.
    impulse_prob_dist : float
        Wasserstein distance between target and generated ditributions
        of estimated impulse probability.
    amp_ratio_dist : float
        Wasserstein distance between target and generated distributions
        of estimated scale ratio.

    """
    targ_prob_ests, gen_prob_ests = [], []
    targ_amp_ratios, gen_amp_ratios = [], []
    i = 0
    for gen_sample, targ_sample in zip(gen_data, targ_data):
        if i % (len(gen_data) // 20) == 0:
            print(i)
        targ_prob_est, targ_sig0_est, targ_sig1_est = estimate_bg_parameters(targ_sample)
        gen_prob_est, gen_sig0_est, gen_sig1_est = estimate_bg_parameters(gen_sample)
        gen_amp_ratio_est = gen_sig1_est / gen_sig0_est
        targ_amp_ratio_est = targ_sig1_est / targ_sig0_est

        targ_prob_ests.append(targ_prob_est)
        gen_prob_ests.append(gen_prob_est)
        targ_amp_ratios.append(targ_amp_ratio_est)
        gen_amp_ratios.append(gen_amp_ratio_est)
        i += 1

    targ_prob_median = np.median(np.array(targ_prob_ests), axis=0)
    gen_prob_median = np.median(np.array(gen_prob_ests), axis=0)
    targ_amp_ratio_median = np.median(np.array(targ_amp_ratios), axis=0)
    gen_amp_ratio_median = np.median(np.array(gen_amp_ratios), axis=0)

    param_dists = {"target": targ_prob_ests, "generated": gen_prob_ests}
    with open(os.path.join(output_path, 'parameter_value_distributions.json'), 'w') as f:
        json.dump(param_dists, f)

    param_dists = {"target": targ_amp_ratios, "generated": gen_amp_ratios}
    with open(os.path.join(output_path, 'parameter_amp_distributions.json'), 'w') as f:
        json.dump(param_dists, f)

    # get event rate earth movers distance:
    impulse_prob_dist = scipy.stats.wasserstein_distance(targ_prob_ests, gen_prob_ests)
    amp_ratio_dist = scipy.stats.wasserstein_distance(targ_amp_ratios, gen_amp_ratios)

    bins = np.histogram(np.hstack((gen_prob_ests, targ_prob_ests)), bins=50)[1]  # get the bin edges
    plt.hist(gen_prob_ests, bins=bins, color="blue", alpha=0.7, label="Generated Distribution")
    plt.hist(targ_prob_ests, bins=bins, color="green", alpha=0.7, label="Target Distribution")
    if param_value is not None:
        plt.axvline(x=param_value, color="black", alpha=0.8, label="True Impulse Probability")
    plt.legend()
    plt.grid()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "impulse_prob.png"), dpi=300)
    plt.clf()

    bins = np.histogram(np.hstack((gen_amp_ratios, targ_amp_ratios)), bins=50)[1]  # get the bin edges
    plt.hist(gen_amp_ratios, bins=bins, color="blue", alpha=0.7, label="Generated Distribution")
    plt.hist(targ_amp_ratios, bins=bins, color="green", alpha=0.7, label="Target Distribution")
    plt.xlabel("Impulse Amplitude Ratio")
    plt.legend()
    plt.grid()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "amplitude_ratio.png"), dpi=300)
    plt.clf()

    return targ_prob_median, gen_prob_median, targ_amp_ratio_median, gen_amp_ratio_median, \
           impulse_prob_dist, amp_ratio_dist

def evaluate_sas_noise(targ_data, gen_data, param_value, output_path):
    """
    Evaluate symmetric alpha stable (SAS) noise dataset

    Parameters
    ----------
    targ_data : ndarray
        Target data.
    gen_data : ndarray
        Generated data.
    param_value : float
        Target characteristic exponent.
    output_path : string
        Directory to save results.

    Returns
    -------
    targ_alpha_median : array
        Median estimated characteric exponent for target data.
    gen_alpha_median : array
        Median estimated characteristic exponent for generated data.
    alpha_dist : float
        Wasserstein distance between target and generated distributions
        of estimated characteristic exponent.

    """
    print("Evaluating generated SAS noise")
    targ_alpha_ests, gen_alpha_ests = [], []
    targ_gamma_ests, gen_gamma_ests = [], []
    i = 0
    for gen_sample, targ_sample in zip(gen_data, targ_data):
        targ_alpha_est = estimate_alpha_fast(targ_sample)
        gen_alpha_est = estimate_alpha_fast(gen_sample)
        gen_gamma_est, _ = estimate_scale_fast(gen_sample, gen_alpha_est)
        targ_gamma_est, _ = estimate_scale_fast(targ_sample, targ_alpha_est)
        targ_alpha_ests.append(targ_alpha_est)
        gen_alpha_ests.append(gen_alpha_est)
        targ_gamma_ests.append(targ_gamma_est)
        gen_gamma_ests.append(gen_gamma_est)
        i += 1

    targ_alpha_median = np.median(np.array(targ_alpha_ests), axis=0)
    gen_alpha_median = np.median(np.array(gen_alpha_ests), axis=0)

    param_dists = {"target": targ_alpha_ests, "generated": gen_alpha_ests}
    with open(os.path.join(output_path, 'parameter_value_distributions.json'), 'w') as f:
        json.dump(param_dists, f)

    param_dists = {"target": targ_gamma_ests, "generated": gen_gamma_ests}
    with open(os.path.join(output_path, 'parameter_scale_distributions.json'), 'w') as f:
        json.dump(param_dists, f)

    gen_alpha_ests = [x for x in gen_alpha_ests if not np.isnan(x)]
    targ_alpha_ests = targ_alpha_ests[:len(gen_alpha_ests)]
    alpha_dist = scipy.stats.wasserstein_distance(targ_alpha_ests, gen_alpha_ests)
    bins = np.histogram(np.hstack((gen_alpha_ests, targ_alpha_ests)), bins=50)[1]  # get the bin edges
    plt.hist(gen_alpha_ests, bins=bins, color="blue", alpha=0.7, label="Generated Distribution")
    plt.hist(targ_alpha_ests, bins=bins, color="green", alpha=0.7, label="Target Distribution")
    if param_value is not None:
        plt.axvline(x=param_value, color="black", alpha=0.8, label="True Alpha Value")
    plt.legend()
    plt.grid()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "alpha_ests.png"), dpi=300)
    plt.clf()
    return targ_alpha_median, gen_alpha_median, alpha_dist

def eval_psd_distances(targ_data, gen_data, output_path):
    """
    Estimate median power spectral densities (PSDs) of target and generated
    distributions and geodesic PSD distance.

    Parameters
    ----------
    targ_data : ndarray
        Target data.
    gen_data : ndarray
        Generated data.
    output_path : string
        Directory to save results.

    Returns
    -------
    median_psd_dist : float
        Estimated geodesic distance between median estimated PSDs for target and generated data.
    """

    targ_PSD_estimates, gen_PSD_estimates = [], []
    for i, (targ_sample, gen_sample) in enumerate(zip(targ_data, gen_data)):
        Sk, weights, _eigenvalues = pmtm(targ_sample, NW=4, k=7, method='eigen')
        Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
        Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
        targ_PSD_estimates.append(Pxx)
        Sk, weights, _eigenvalues = pmtm(gen_sample, NW=4, k=7, method='eigen')
        Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
        Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
        gen_PSD_estimates.append(Pxx)

    targ_median_psd = np.median(np.array(targ_PSD_estimates), axis=0)
    gen_median_psd = np.median(np.array(gen_PSD_estimates), axis=0)
    targ_max_psd = np.amax(np.array(targ_PSD_estimates), axis=0)
    gen_max_psd = np.amax(np.array(gen_PSD_estimates), axis=0)
    targ_min_psd = np.amin(np.array(targ_PSD_estimates), axis=0)
    gen_min_psd = np.amin(np.array(gen_PSD_estimates), axis=0)

    h5f = h5py.File(os.path.join(output_path,'median_psds.h5'), "w")
    h5f.create_dataset('gen_median_psd', data=gen_median_psd)
    h5f.create_dataset('targ_median_psd', data=targ_median_psd)
    h5f.create_dataset('gen_max_psd', data=gen_max_psd)
    h5f.create_dataset('targ_max_psd', data=targ_max_psd)
    h5f.create_dataset('gen_min_psd', data=gen_min_psd)
    h5f.create_dataset('targ_min_psd', data=targ_min_psd)
    h5f.close()
    w = np.linspace(0, 0.5, len(gen_median_psd))
    plt.plot(w, 10 * np.log10(targ_median_psd), color='blue', alpha=0.75, label='Target')
    plt.plot(w, 10 * np.log10(gen_median_psd), color='green', alpha=0.75, label='Generated')
    plt.ylabel('Power Density (dB)')
    plt.xlabel("Normalized Digital Frequency (cycles/sample)")
    plt.grid()
    plt.margins(x=0)
    plt.legend()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "median_psd_comparison.png"), dpi=300)
    plt.clf()

    # calculate PSD distance estimates
    step_size = 0.5 / len(gen_median_psd)
    # distance between median psds
    log_med_psd_ratio = np.log(gen_median_psd) - np.log(targ_median_psd)
    median_psd_dist = np.sqrt(np.sum(log_med_psd_ratio ** 2 * step_size / 0.5)
                              - np.sum(log_med_psd_ratio * step_size / 0.5) ** 2)

    return median_psd_dist

