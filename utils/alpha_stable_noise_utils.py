#!/usr/bin/env python3
"""
Methods for simulating symmetric alpha stable (sas) processes and
estimating parameters.

Author: Adam Wunderlich
Date: June 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from spectrum import pmtm
from scipy import stats
from scipy.special import gamma
from numpy import pi
import levy  # available at https://github.com/josemiotto/pylevy
rng = default_rng()


def simulate_sas_noise(signal_length, alpha):
    """
    Simulate symmetric alpha stable noise with zero skewness (beta),
    zero location (delta or mu), unit scale (gamma or sigma), under
    parameterization '0' from (Nolan, "Univariate Stable Distributions", Springer, 2020).

    Parameters
    ----------
    signal_length : int
        length of time series.
    alpha : float
        characteristic exponent, accepts values between 0.5 and 2.

    Returns
    -------
    X : 1-D numpy array
        SAS noise process.

    """

    X = levy.random(alpha, beta=0, shape = signal_length)
    return X

def estimate_alpha_fast(X):
    """
    Fast estimation of characteristic exponent (alpha)
    for a SAS process using method of (Tsihrintzis and Nikias, 1996)

    Parameters
    ----------
    X : 1-D numpy array with length divisible by 128
        SAS time series.

    Returns
    -------
    alpha_est : float
        estimate of characteristic exponent.
    """

    L = 128 # number of segments for time-series
    K = int(len(X)/L) # length of each segment
    assert len(X) % L ==0, 'signal length not divisible by L'
    delta_hat = np.median(X)
    X = X-delta_hat
    Xseg = X.reshape(L,K)
    Xmin = -np.log(-Xseg.min(axis=1))
    Xmax = np.log(Xseg.max(axis=1))
    s_min = np.std(Xmin,ddof=1)
    s_max = np.std(Xmax,ddof=1)
    alpha_hat = pi/(2*np.sqrt(6))*(1/s_max + 1/s_min)
    return alpha_hat

def estimate_scale_fast(X,alpha_hat):
    """
    Fast estimation of scale (dispersion) and location parameters for a
    SAS process using method of (Tsihrintzis and Nikias, 1996).

    Parameters
    ----------
    X :  1-D numpy array
        SAS time series.
    alpha_hat : float
        estimate of alpha, the characteristic exponent.

    Returns
    -------
    gamma_hat : float
        estimate of scale parameter.
    delta_hat : float
        estimate of location parameter.
    """

    delta_hat = np.median(X)
    Xcen = X - delta_hat
    p = alpha_hat/3 # recommended value; see footnote #1 in (Tsihrintzis and Nikias,1996)
    numer = np.mean(np.abs(Xcen)**p)
    C = 1/np.cos(pi*p/2)*gamma(1-p/alpha_hat)/gamma(1-p)
    gamma_hat = (numer/C)**(alpha_hat/p)
    return gamma_hat, delta_hat


def estimate_params_ML(X):
    """
    Estimate parameters of SAS distribution using ML estimators.
    Requires alpha>=0.5

    Parameters
    ----------
    X: 1-D numpy array
        SAS process.

    Returns
    -------
    alpha_est: float
        estimate of characteristic exponent.
    gamma_est: float
        estimate of scale parameter.
    """

    params = levy.fit_levy(X, beta=0.0, mu=0.0)
    alpha_est = params[0].get('1')[0]
    gamma_est = params[0].get('1')[3]
    return alpha_est, gamma_est

def sas_pdf(x, alpha):
    """
    Calculate probability density function (PDF) for standard SAS distribution.

    Parameters
    ----------
    x: 1-D numpy array
        values where PDF is evaluated
    alpha: float
        characteristic exponent in the range [0,2)

    Returns
    -------
    f : 1-D numpy array
        PDF values
    """

    f = levy.levy(x, alpha, beta=0.0, mu=0.0, sigma=1.0)
    return f

def estimate_psd(X):
    """
    Estimate power spectral density using the multitaper method.

    Parameters
    ----------
    X: 1-D numpy array
        time-series.

    Returns
    -------
    Pxx : numpy array
        PSD estimate.
    w : numpy array
        normalized digital frequencies.
    """

    Sk, weights, _eigenvalues = pmtm(X, NW=4, k=7, method='eigen')
    Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
    Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
    w = np.linspace(0, 0.5, len(Pxx))  # normalized digital frequency
    return Pxx, w

def main():
    # testing and examples

    signal_length = 4096
    alpha = np.linspace(.5,1.5,6)
    num_repeats = 500

    alpha_est = np.zeros((len(alpha), num_repeats))
    gamma_est = np.zeros((len(alpha), num_repeats))
    delta_est = np.zeros((len(alpha), num_repeats))
    X_examples = np.zeros((len(alpha), signal_length))
    median_psd = np.zeros((len(alpha), int(signal_length/2)))
    for k in np.arange(len(alpha)):
        print(f"alpha {k+1} of {len(alpha)}")
        psd_list = []
        for n in np.arange(num_repeats):
            X = simulate_sas_noise(signal_length,alpha[k])
            alpha_est[k,n] = estimate_alpha_fast(X)
            gamma_est[k,n], delta_est[k,n] = estimate_scale_fast(X, alpha_est[k,n])
            #alpha_est[k,n], gamma_est[k,n] = estimate_params_ML(X)
            Pxx, w, = estimate_psd(X)
            psd_list.append(Pxx)
        X_examples[k,:] = X
        median_psd[k,:] = np.median(np.array(psd_list), axis=0)
    true_alpha = np.repeat(alpha, num_repeats)
    median_alpha_est = np.median(alpha_est, axis=1)
    iqr_alpha_est = stats.iqr(alpha_est, axis=1)
    median_gamma_est = np.median(gamma_est, axis=1)
    iqr_gamma_est = stats.iqr(gamma_est, axis=1)
    median_delta_est = np.median(delta_est, axis=1)
    iqr_delta_est = stats.iqr(delta_est, axis=1)

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(true_alpha, alpha_est.flatten(), 'o', alpha = 0.05, label = 'alpha estimates')
    ax1.plot(alpha, median_alpha_est, 'x', color='black', label = 'median estimates')
    ax1.plot(alpha, alpha,'r', label = 'truth')
    ax1.set_ylabel('estimated alpha')
    ax1.set_xlabel('true alpha')
    ax1.legend()
    ax1.grid()
    ax1.set_title('alpha scatterplot')
    ax2.plot(alpha,(median_alpha_est-alpha)/alpha*100, 'o', label='relative bias')
    ax2.plot(alpha, iqr_alpha_est/alpha*100, 'ro', label='relative IQR')
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('error (%)')
    ax2.grid()
    ax2.legend()
    ax2.set_title('alpha accuracy')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(true_alpha, gamma_est.flatten(), 'o', alpha = 0.05, label = 'gamma estimates')
    ax1.plot(alpha, median_gamma_est, 'x', color='black', label = 'median estimates')
    ax1.plot(alpha, np.ones(len(alpha)),'r', label = 'truth')
    ax1.set_ylabel('estimated gamma')
    ax1.set_xlabel('alpha')
    ax1.legend()
    ax1.grid()
    ax1.set_title('gamma scatterplot')
    ax2.plot(alpha,(median_gamma_est-np.ones(len(alpha)))/1*100, 'o', label='relative bias')
    ax2.plot(alpha, iqr_gamma_est/1*100, 'ro', label='relative IQR')
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('error (%)')
    ax2.grid()
    ax2.legend()
    ax2.set_title('gamma accuracy')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    """
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(true_alpha, delta_est.flatten(), 'o', alpha = 0.05, label = 'delta estimates')
    ax1.plot(alpha, median_delta_est, 'x', color='black', label = 'median estimates')
    ax1.plot(alpha, np.zeros(len(alpha)),'r', label = 'truth')
    ax1.set_ylabel('estimated delta')
    ax1.set_xlabel('alpha')
    ax1.legend()
    ax1.grid()
    ax1.set_title('delta scatterplot')
    ax2.plot(alpha,(median_delta_est-np.zeros(len(alpha)))*100, 'o', label='bias')
    ax2.plot(alpha, iqr_delta_est, 'ro', label='IQR')
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('error (%)')
    ax2.grid()
    ax2.legend()
    ax2.set_title('delta accuracy')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()
    """

    """
    fig4, ax = plt.subplots(1, 1, figsize=(6, 6))
    for k in np.arange(len(alpha)):
        ax.plot(w, 10*np.log10(median_psd[k,:]), label = f'median PSD, alpha={alpha[k]}')
    ax.set_ylabel('Power Density (dB)')
    ax.set_xlabel('Normalized Frequency')
    ax.grid()
    ax.legend()
    ax.set_title('PSD comparison')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    fig3, ax_array = plt.subplots(len(alpha), 1, figsize=(10, 7))
    for k, ax in enumerate(ax_array):
        ax.plot(X_examples[k,:])
        ax.set_title(f'example time-series, alpha = {alpha[k]}')
    plt.tight_layout(pad=0.25, w_pad=1, h_pad=1)
    plt.show()
    """

if __name__ == "__main__":
    main()
