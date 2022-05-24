#!/usr/bin/env python3
''' Methods for generating Bernoulli-Gaussian _archive processes and
    estimating parameters.
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from spectrum import pmtm
from sklearn.mixture import GaussianMixture
from scipy import stats
rng = default_rng()

def simulate_bg_noise(signal_length, p, sig_w, sig_i):
    n_w = rng.standard_normal(signal_length)*sig_w
    n_i = rng.binomial(1,p,signal_length)*rng.standard_normal(signal_length)*sig_i
    X = n_w+n_i
    return X

def estimate_bg_parameters(X):
    if X.ndim == 1:
        X = X.reshape(-1,1)
    gm = GaussianMixture(n_components=2, tol = 1e-6, max_iter = 500,
                         n_init = 1, means_init = np.zeros((2,1)),
                         random_state=0).fit(X)
    sig0 = np.min(np.sqrt(gm.covariances_))
    sig1 = np.max(np.sqrt(gm.covariances_))
    ind = np.argmax(np.sqrt(gm.covariances_))
    p = gm.weights_[ind]
    return p, sig0, sig1

def bg_pdf(x, p, sig_w, sig_i):
    # return PDF values for the BG mixture distribution
    f_BG = ((1-p)*stats.norm.pdf(x, loc=0, scale=sig_w)
            + p*stats.norm.pdf(x, loc=0, scale=np.sqrt(sig_w**2 + sig_i**2)))
    return f_BG

def estimate_psd(X):
    Sk, weights, _eigenvalues = pmtm(X, NW=4, k=7, method='eigen')
    Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
    Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
    w = np.linspace(0, 0.5, len(Pxx))  # normalized digital frequency
    return Pxx, w

def main():
    # testing and examples

    signal_length = 4096
    impulse_prob = [.01, .05, .1, .25, .5, .8, .9]
    sig_w = 0.1
    sig_i = 1
    num_repeats = 100

    p_est = np.zeros((len(impulse_prob), num_repeats))
    scale_ratio_est = np.zeros((len(impulse_prob),num_repeats))
    X_examples = np.zeros((len(impulse_prob), signal_length))
    median_psd = np.zeros((len(impulse_prob), int(signal_length/2)))
    for k, p in enumerate(impulse_prob):
        print(f"impulse rate {k+1} of {len(impulse_prob)}")
        psd_list = []
        for n in np.arange(num_repeats):
            X = simulate_bg_noise(signal_length, p, sig_w, sig_i)
            p_est[k, n], sig0_est, sig1_est = estimate_bg_parameters(X)
            scale_ratio_est[k, n] = sig1_est/sig0_est
            Pxx, w, = estimate_psd(X)
            psd_list.append(Pxx)
        X_examples[k, :] = X
        median_psd[k, :] = np.median(np.array(psd_list), axis=0)
    true_p = np.repeat(impulse_prob, num_repeats)
    median_p_est = np.median(p_est, axis=1)
    iqr_p_est = stats.iqr(p_est, axis=1)
    median_scale_ratio_est = np.median(scale_ratio_est, axis=1)
    iqr_scale_ratio_est = stats.iqr(scale_ratio_est, axis=1)
    scale_ratio_truth = np.sqrt(sig_w**2 + sig_i**2)/sig_w

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))
    ax1.plot(true_p, p_est.flatten(), 'o', alpha=0.05, label='p estimates')
    ax1.plot(impulse_prob, median_p_est, 'x', color='black', label='median estimates')
    ax1.plot(impulse_prob, impulse_prob, 'r', label='truth')
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('estimated p')
    ax1.set_xlabel('true p')
    ax1.legend()
    ax1.grid()
    ax2.plot(impulse_prob, (median_p_est - impulse_prob)/ impulse_prob * 100, 'o', label='relative bias')
    ax2.plot(impulse_prob, iqr_p_est/impulse_prob * 100, 'ro', label='relative IQR')
    ax2.set_xlabel('p')
    ax2.set_ylabel('error (%)')
    ax2.grid()
    ax2.legend()
    ax3.plot(true_p, scale_ratio_est.flatten(), 'o', alpha = 0.05, label = 'scale ratio estimates')
    ax3.plot(impulse_prob, median_scale_ratio_est, 'x', color='black', label = 'median estimates')
    ax3.hlines(scale_ratio_truth,0,impulse_prob[-1], colors='red', label = 'truth')
    ax3.set_ylabel('scale ratio')
    ax3.set_xlabel('p')
    ax3.legend()
    ax3.grid()
    ax4.set_title('estimated p accuracy')
    ax4.plot(impulse_prob, (median_scale_ratio_est - scale_ratio_truth) / scale_ratio_truth * 100, 'o', label='relative bias')
    ax4.plot(impulse_prob, iqr_scale_ratio_est / scale_ratio_truth * 100, 'ro', label='relative IQR')
    ax4.set_xlabel('p')
    ax4.set_ylabel('error (%)')
    ax4.grid()
    ax4.legend()
    ax4.set_title('estimated scale ratio accuracy')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    fig2, ax = plt.subplots(1, 1, figsize=(6, 6))
    for k in np.arange(len(impulse_prob)):
        ax.plot(w, 10*np.log10(median_psd[k,:]), label = f'median PSD, p={impulse_prob[k]}')
    ax.set_ylabel('Power Density (dB)')
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylim([-50,0])
    ax.grid()
    ax.legend()
    ax.set_title('PSD comparison')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    fig3, ax_array = plt.subplots(len(impulse_prob), 1, figsize=(10, 7))
    for k, ax in enumerate(ax_array):
        ax.plot(X_examples[k,:])
        ax.set_title(f'example time-series, p = {impulse_prob[k]}')
    plt.tight_layout(pad=0.25, w_pad=1, h_pad=1)
    plt.show()


if __name__ == "__main__":
    main()
