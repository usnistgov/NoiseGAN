#!/usr/bin/env python3
'''
Methods for generating generalized shot noise processes and estimating the
event rate parameter.

Author: Adam Wunderlich
Date: June 2022
'''


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import signal
from scipy import stats
from spectrum import pmtm
rng = default_rng()


def simulate_shot_noise(signal_length, pulse_type, amp_distrib, event_rate, tau_d, beta, theta):
    """
     Simulate generalize shot _archive (filtered Poisson process)
    :param signal_length:
    :param pulse_type: string specifiying pulse shape
    :param amp_distrib: string specifying amplitude distribution
    :param event_rate: events per unit time
    :param tau_d: pulse duration
    :param beta: mean pulse amplitude
    :param theta: normalized time step
    :return: generalized shot _archive process
    """
    warmup = signal_length
    sig_len = signal_length + warmup
    delta_t = theta * tau_d
    T = (sig_len - 1) * delta_t
    K = rng.poisson(event_rate * T)  # number of pulses in time interval
    if amp_distrib == 'exponential':
        A = rng.exponential(beta, K)  # random pulse amplitudes
    elif amp_distrib == 'rayleigh':
        scale = beta*np.sqrt(2 / np.pi)
        A = rng.rayleigh(scale, K),
    elif amp_distrib == 'standard_normal':
        A = rng.standard_normal(K)
    mk = rng.integers(0, sig_len, K)  # indices for arrival times
    f_K = np.zeros(sig_len)
    for k in np.arange(0, K):
        f_K[mk[k]] = A[k]
    pulse, t, I1, I2 = pulse_function(sig_len, pulse_type, theta, tau_d)
    X = signal.fftconvolve(f_K, pulse, 'same')
    X = X[warmup:]
    return X


def pulse_function(signal_length, pulse_type, theta, tau_d):
    # Outputs: pulse (pulse waveform), t (time),
    #          I1, I2 (first and second integrals of pulse)
    """
    Compute pulse functions for generalized shot noise
    :param signal_length:
    :param pulse_type:
    :param theta:
    :param tau_d:
    :return: pulse (pulse waveform), t (time values, centered at zero),
             I1, I2 (first and second integrals of pulse)
    """

    delta_t = theta*tau_d
    m = np.arange(0, signal_length)
    t = (m - signal_length/2)*delta_t  # time values, centered at zero
    pos_ind = np.where(t >= 0)
    pulse = np.zeros(signal_length)
    if pulse_type == 'one_sided_exponential':
        pulse[pos_ind] = (1/tau_d)*np.exp(-t[pos_ind]/tau_d)
        I1 = 1
        I2 = 1/(2*tau_d)
    elif pulse_type == 'linear_exponential':
        pulse[pos_ind] = t[pos_ind]/(tau_d**2)*np.exp(-t[pos_ind]/tau_d)
        I1 = 1
        I2 = 1/(4*tau_d)
    elif pulse_type == 'gaussian':
        pulse = np.exp(-(t**2)/(2*tau_d**2))/(np.sqrt(2*np.pi)*tau_d)
        I1 = 1
        I2 = 1/(2*tau_d*np.sqrt(np.pi))
    return pulse, t, I1, I2


def estimate_event_rate(X, pulse_type, amp_distrib, theta, tau_d):
    """
    Estimate event rate of filtered Possion process
    :param X: FP process
    :param pulse_type:
    :param amp_distrib:
    :param theta:
    :param tau_d:
    :return: estimated event rate
    """
    signal_length = len(X)
    pulse, t, I1, I2 = pulse_function(signal_length, pulse_type, theta, tau_d)
    mean_X = np.mean(X)
    var_X = np.var(X)
    delta_t = theta*tau_d
    T = (signal_length-1)*delta_t
    if amp_distrib == 'exponential':
        nu_est = 2*mean_X**2*I2/(var_X*I1**2)  # T >> tau_d
        # nu_est = (mean_X**2/var_X)*(2*I2/(I1**2) - 1/T)
    elif amp_distrib == 'rayleigh':
        nu_est = 4*mean_X**2*I2/(var_X*np.pi*I1**2)  # T >> tau_d
        # nu_est = (mean_X**2/var_X)*(4*I2/(np.pi*I1**2) - 1/T)
        # Note: the rayleigh estimators exhibit increasing relative bias with increasing nu
    elif amp_distrib == 'standard_normal':
        nu_est = var_X/I2
    return nu_est


def estimate_acf(X, pulse_type, event_rate, theta, tau_d, beta, amp_distrib):
    """
    return estimated and theoretical normalized PSD and ACF

    :param X:
    :param pulse_type:
    :param event_rate:
    :param theta:
    :param tau_d:
    :param beta:
    :param amp_distrib:
    :return:
    """
    signal_length = len(X)
    pulse, t, I1, I2 = pulse_function(signal_length, pulse_type, theta, tau_d)
    X = X - np.mean(X)  # remove mean

    # estimate PSD
    # Sk, weights, _eigenvalues = pmtm(X, NW=4, k=7, method='eigen')
    # Pxx = np.abs(np.mean(Sk * weights, axis=0)) ** 2
    # Pxx = Pxx[0: (len(Pxx) // 2)]  # one-sided psd
    # w = np.linspace(0, 1, len(Pxx))  # normalized digital frequency
    # Pxx = Pxx / Pxx[0]  # normalize

    # estimate ACF
    Rxx = np.correlate(X, X, mode='full')
    midpt = int((len(Rxx) + 1) / 2)
    Rxx = Rxx[midpt:]  # one-sided ACF
    tau = np.arange(0, int(6 / theta))  # lags
    Rxx = Rxx[tau]
    Rxx = Rxx / Rxx[0]  # normalize

    if amp_distrib == 'exponential':
        amp_mean = beta
        amp_rawmom2 = 2 * beta ** 2
        amp_cenmom2 = beta ** 2
    elif amp_distrib == 'rayleigh':
        scale = beta*np.sqrt(2 / np.pi)
        amp_mean = beta
        amp_rawmom2 = 2 * scale ** 2
        amp_cenmom2 = ((4 - np.pi) / 2) * scale ** 2
    elif amp_distrib == 'standard_normal':
        amp_mean = 0
        amp_rawmom2 = 1
        amp_cenmom2 = 1

    # theoretical ACF
    Rxx_theory = event_rate * amp_rawmom2 * np.correlate(pulse, pulse, mode='full')
    midpt = int((len(Rxx_theory) + 1) / 2)
    Rxx_theory = Rxx_theory[midpt:]  # one-sided ACF
    tau = np.arange(0, int(6 / theta))  # lags
    Rxx_theory = Rxx_theory[tau]
    Rxx_theory = Rxx_theory / Rxx_theory[0]  # normalize

    # theoretical PSD
    # Pxx_theory = event_rate * (amp_mean ** 2 + amp_cenmom2) * np.abs(np.fft.fft(pulse)) ** 2
    # midpt = int((len(Pxx_theory) + 1) / 2)
    # Pxx_theory = Pxx_theory[0: midpt]  # one-sided PSD
    # wt = np.linspace(0, 1, len(Pxx_theory))  # normalized digital frequency
    # Pxx_theory = Pxx_theory / Pxx_theory[0]  # normalize
    return Rxx, tau, Rxx_theory


def main():
    # testing and examples
    signal_length = 4096
    pulse_type_list = ['one_sided_exponential', 'linear_exponential', 'gaussian']
    pulse_type = pulse_type_list[0]
    amp_distrib = 'exponential' # 'standard_normal', 'exponential', 'rayleigh'
    tau_d, beta, theta = 1, 1, 0.1
    event_rate = np.arange(.25, 3.25, .25)
    num_repeats = 500  # number of repeats at each event rate

    nu_est = np.zeros((len(event_rate), num_repeats))
    mu_est = np.zeros((len(event_rate), num_repeats))
    mu_theory = np.zeros(len(event_rate))
    for k in np.arange(0, len(event_rate)):
        print(f"event rate {k} of {len(event_rate)}")
        acf_list = []
        for n in np.arange(num_repeats):
            X = simulate_shot_noise(signal_length, pulse_type, amp_distrib, event_rate[k], tau_d, beta, theta)
            nu_est[k, n] = estimate_event_rate(X, pulse_type, amp_distrib, theta, tau_d)
            mu_est[k,n] = np.mean(X)
            if k == 0 or k == len(event_rate)-1:
               Rxx, tau, Rxx_theory = estimate_acf(X, pulse_type, event_rate[k], theta, tau_d, beta, amp_distrib)
               acf_list.append(Rxx)
        if k == 0:
            X1 = X
            median_acf1 = np.median(np.array(acf_list), axis=0)
            acf1_theory = Rxx_theory
        elif k == len(event_rate)-1:
            X2 = X
            median_acf2 = np.median(np.array(acf_list), axis=0)
            acf2_theory = Rxx_theory
        mu_theory[k] = event_rate[k]*beta
    nus = np.repeat(event_rate, num_repeats)
    median_nu_est = np.median(nu_est, axis=1)
    mean_nu_est = np.mean(nu_est, axis=1)
    iqr_nu_est = stats.iqr(nu_est, axis=1)
    mu_est_mean = np.mean(mu_est, axis=1)



    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))
    ax1.plot(nus, nu_est.flatten(), 'o', alpha = 0.05, label = 'nu estimates')
    ax1.plot(event_rate, median_nu_est, 'x', color='black', label = 'median estimates')
    ax1.plot(event_rate,event_rate,'r', label = 'truth')
    ax1.set_ylim([0,10])
    ax1.set_ylabel('estimated nu')
    ax1.set_xlabel('true nu')
    ax1.set_title('estimator evaluation')
    ax1.legend()
    ax1.grid()
    ax2.plot(event_rate,(mean_nu_est-event_rate)/event_rate, 'o')
    ax2.set_xlabel('nu')
    ax2.set_ylabel('relative bias')
    ax2.grid()
    ax3.plot(event_rate, iqr_nu_est/event_rate, 'o')
    ax3.set_xlabel('nu')
    ax3.set_ylabel('IQR/nu')
    ax3.grid()
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(10, 7))
    ax4.plot(X1)
    ax4.set_title(f'example time-series, nu = {event_rate[0]}')
    ax5.plot(X2)
    ax5.set_title(f'example time-series, nu = {event_rate[-1]}')
    ax6.plot(event_rate,mu_est_mean, 'o', color= 'blue', label='estimated')
    ax6.plot(event_rate,mu_theory, 'x', color='red', label='theory')
    ax6.set_title('mean comparison')
    ax6.grid()
    ax6.legend()
    ax7.plot(tau, median_acf1, color = 'blue', label = f'median ACF, nu={event_rate[0]}')
    ax7.plot(tau, acf1_theory, '--', color = 'black', label = f'theory, nu={event_rate[0]}')
    ax7.plot(tau, median_acf2, color = 'green', label = f'median ACF, nu={event_rate[-1]}')
    ax7.plot(tau, acf2_theory, '--', color = 'black', label = f'theory, nu={event_rate[-1]}')
    ax7.grid()
    ax7.legend()
    ax7.set_title('normalized ACF comparison')
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

    fig3, (ax8, ax9) = plt.subplots(1, 2, figsize=(11, 4))
    pt_ind  = np.concatenate((signal_length/2 - np.flip(tau),
                              signal_length/2 + tau))
    pt_ind = pt_ind.astype(int)
    for k in range(3):
        pulse, t, I1, I2 = pulse_function(signal_length, pulse_type_list[k],
                                          theta, tau_d)
        Rxx, tau, Rxx_theory = estimate_acf(X, pulse_type_list[k], event_rate[0], theta, tau_d, beta, amp_distrib)
        ax8.plot(t[pt_ind], pulse[pt_ind])
        ax9.plot(tau, Rxx_theory)
    ax8.grid()
    ax8.set_title('pulse types')
    ax8.set_xlabel('time')
    ax8.set_ylabel('amplitude')
    ax9.set_title('normalized theoretical ACF')
    ax9.set_xlabel('lag')
    ax9.set_ylabel('Autocovariance')
    ax9.grid()
    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    plt.show()

if __name__ == "__main__":
    main()
