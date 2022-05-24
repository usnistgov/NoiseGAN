#!/usr/bin/env python3
"""
Module for generating and estimating parameters for fractional _archive random process models, including
wide-sense stationary fractional Gaussian _archive (FGN), fractionally differenced white _archive (FDWN), and
nonstationary fractional Brownian motion (FBM).

Author: Adam Wunderlich Jan 2021
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from contextlib import contextmanager
from numpy.random import default_rng
from rpy2.robjects.packages import importr
from rpy2 import rinterface as ri
from rpy2 import robjects
# suppress R warnings
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR) 

ri.initr()  # initialize low-level R interface
arfima = importr('arfima')  # import arfima package from R
rng = default_rng()  # random number generator object


def simulate_FGN(signal_length, H, sigma_sq=1):
    """
    Simulate fractional Gaussian _archive (FGN): Hurst exponent is in the range (0,1) (H < 0.5 yields
    anti-persistent (short-memory) FGN,  H=0.5 corresponds to WGN, H > 0.5 yields persistent (long-memory) FGN).

    References:
        1) Perrin et al, "Fast and Exact Synthesis for 1-D Fractional Brownian Motion and Fractional Gaussian Noises,"
        IEEE Signal Processing Letters, 9(11), pp. 282-284, Nov 2002.
        2) Dietrich and Newsam, "Fast and exact simulation of stationary Gaussian processes through circulant embedding
        of the covariance matrix," SIAM Journal on Scientific Computing, 18(4), pp.1088-1107, 1997.

    :param signal_length:
    :param H: Hurst exponent
    :param sigma_sq: variance of FGN
    :return: Gh1, Gh2 (two independent fractional Gaussian _archive realizations)
    """
    M = signal_length - 1  # fft length is 2M
    k1 = np.arange(signal_length)
    k2 = np.arange(1, signal_length - 1)
    s = np.zeros(2 * M)
    s[k1] = FGNacorr(H, k1, sigma_sq)
    s[2 * M - k2] = FGNacorr(H, k2, sigma_sq)
    s_tilde = np.fft.fft(s)
    x = rng.normal(0, 1, 2 * M) + 1j * rng.normal(0, 1, 2 * M)
    x_tilde = np.sqrt(s_tilde / (2 * M)) * x
    y = np.fft.fft(x_tilde)
    # fractional Gaussian _archive realizations
    Gh1, Gh2 = np.real(y[0: signal_length]), np.imag(y[0: signal_length])
    return Gh1, Gh2


def FGNacorr(H, k, sigma_sq=1):
    """
    Autocorrelation function for fractional Gaussian _archive
    :param H: Huerst exponent
    :param k:
    :param sigma_sq: variance of FGN
    :return:
    """
    rG = (sigma_sq / 2) * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    return rG


def FGN_to_FBM(FGN_time_series):
    """
    Convert fractional Gaussian _archive to a fractional Brownian motion process
    :param FGN_time_series:
    :return:
    """
    N = len(FGN_time_series)
    FBM_time_series = np.concatenate((np.array([0]), np.cumsum(FGN_time_series[1: N])))
    return FBM_time_series


def estimate_Hurst_exponent(x, process_type='FGN'):
    """
    Estimate Hurst exponent for fractional Gaussian _archive or fractional Brownian motion.

    Implements the discrete variations estimator (Istas & Lang, 1997), (Coeurjolly, 2001), (Courjolly & Porcu,2017)
    using the 2nd-order difference. This code is based on estCFBM.R by (Coeurjolly,2017) and the Matlab function
    wfbmesti.m (Mathworks, 2003).  When M2=2, it is equivalent to the first method of wfbmesti.m.

    References:
        1) Istas & Lang, "Quadratic variations and estimation of the local Hölder
        index of a Gaussian process,"Annales de l'Institut Henri Poincare (B)
        Probability and Statistics, vol. 33, no. 4, pp. 407-436. 1997.
        2) Coeurjolly, "Estimating the parameters of a fractional Brownian motion
        by discrete variations of its sample paths," Statistical Inference for
        Stochastic Processes, 4(2), 199-227, 2001.
        3) Coeurjolly & Porcu, "Properties and Hurst exponent estimation of the
        circularly-symmetric fractional Brownian motion,"
        Statistics & Probability Letters, vol. 128, pp. 21-27, 2017.

    :param x: real or complex-valued FGN or FBM process
    :param process_type: 'FGN' or 'FBM'
    :return: Estimated Hurst exponent
    """
    b = np.array([1, -2, 1])  # 2nd-order difference filter
    Lb = len(b)
    M1 = 1  # smallest filter dilation
    M2 = 5  # largest filter dilation
    if process_type == 'FBM':
        x = np.diff(x)
    Z = FGN_to_FBM(x)
    L = len(Z)
    U = np.zeros(M2 - M1 + 1)
    for m in np.arange(M1, M2 + 1):
        # upsample filter by dilation factor, m
        bm = np.zeros(Lb * m)
        bm[::m] = b  # dilated filter
        if np.all(np.isreal(Z)):
            yf = signal.lfilter(bm, 1, Z)
            y = yf[(len(bm)-1):L]  # extract causal part of convolution
        else:
            yfr = signal.lfilter(bm, 1, np.real(Z))
            yfi = signal.lfilter(bm, 1, np.imag(Z))
            yf = yfr + 1j*yfi
            y = yf[(len(bm)-1): L]  # extract causal part of convolution
        U[m - 1] = np.mean(np.abs(y)**2)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(np.log(np.arange(M1, M2 + 1)), np.log(U))
    Hest = slope/2
    return Hest


@contextmanager
def suppress_stdout():
    # function to suppress unwanted console output 
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def simulate_FDWN(signal_length, dfrac):
    """
    Simulate FDWN time-series
    :param signal_length:
    :param dfrac: fractional _archive parameter in the range (-.5,.5)
    :return: output: FDWN_time_series
    """
    arfima_sim = robjects.r['arfima.sim']
    model = robjects.r.list(dfrac=ri.FloatSexpVector((dfrac,)))
    FDWN_time_series = arfima_sim(signal_length, model)  # simulate FDWN process
    return FDWN_time_series


def estimate_FD_param(x):
    """
    Fit fractionally-differenced white _archive model
    :param x: real-valued time-series
    :return: dfrac (estimated fractional _archive parameter),
             dfrac_std (estimated standard error on dfrac)
    """
    x = robjects.vectors.FloatSexpVector(x)
    order = robjects.r.c(0, 0, 0)  # ARIMA model order
    with suppress_stdout():
        fit = robjects.r['arfima'](x, order, lmodel='d')
        modes = robjects.r.unlist(fit.rx2('modes'))
        dfrac = np.array(modes.rx2('dfrac'))[0]
        dfrac_std = np.array(modes.rx2('se1'))[0]
    return dfrac, dfrac_std


def main():
    # testing and examples
    example = 'FDWN' # 'FGN' or 'FDWN'
            
    if example == 'FGN':     
        signal_length = 2048
        Hvec = np.arange(.05, 1, .05) # vector of true Hurst exponent values
        Nsim = len(Hvec)
        Hest = np.zeros(Nsim)
        for k,H_true in enumerate(Hvec): 
            if k%10 == 0:
                print('run {0} of {1}'.format(k,Nsim))
            Gh1, Gh2 = simulate_FGN(signal_length, H_true)
            Hest1 = estimate_Hurst_exponent(Gh1, process_type = 'FGN')
            Hest2 = estimate_Hurst_exponent(Gh2, process_type = 'FGN')    
            Hest[k] = (Hest1 + Hest2)/2  # average Hest estimates for two independent realizations  
        fig,ax = plt.subplots(1,1)
        ax.plot(Hvec, Hest, 'o', label = 'H estimates')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_ylabel('estimated Hurst exponent')
        ax.set_xlabel('true Hurst exponent')
        ax.set_title('FGN evaluation')
        ax.plot(Hvec,Hvec,'r', label = 'truth')
        ax.legend()
        ax.grid()
        
        # plot specific time-series examples
        H_true = 0.25
        FGN_time_series1, FGN_time_series2 = simulate_FGN(signal_length, H_true) 
        FBM_time_series1 = FGN_to_FBM(FGN_time_series1)
        FBM_time_series2 = FGN_to_FBM(FGN_time_series2)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(FGN_time_series1, color='blue', label = 'realization 1')
        ax1.plot(FGN_time_series2, color='red', label = 'realization 2')
        ax1.grid()
        title1 = 'FGN examples, H = {:.2}'.format(H_true)
        ax1.set_title(title1)
        ax1.set_xlabel('time index')
        ax1.set_ylabel('amplitude')
        ax1.legend()
        
        ax2.plot(FBM_time_series1, color='blue', label = 'realization 1')
        ax2.plot(FBM_time_series2, color='red', label = 'realization 1')
        ax2.grid()
        title2 = 'FBM examples, H = {:.2}'.format(H_true)
        ax2.set_title(title2)
        ax2.legend()
        ax2.set_xlabel('time index')
        ax2.set_ylabel('amplitude')        
        plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
        
    elif example == 'FDWN':
        signal_length = 2048
        df_vec = np.arange(-.45, .5, .05) # vector of true dfrac parameters
        Nsim = len(df_vec)
        dfrac = np.zeros(Nsim)
        dfrac_std = np.zeros(Nsim)
        for k, df_true in enumerate(df_vec): 
            if k % 10 == 0:
                print('run {0} of {1}'.format(k, Nsim))
            FDWN_time_series = np.array(simulate_FDWN(signal_length, df_true))
            dfrac[k], dfrac_std[k] = estimate_FD_param(FDWN_time_series)          
        fig, ax = plt.subplots(1,1)
        ax.plot(df_vec,dfrac, 'o', label = 'dfrac estimates')
        ax.set_xlim([-.5,.5])
        ax.set_ylim([-.5,.5])
        ax.set_ylabel('estimated dfrac')
        ax.set_xlabel('true dfrac')
        ax.set_title('FDWN evaluation')
        ax.plot(df_vec,df_vec,'r', label = 'truth')
        ax.legend()
        ax.grid()
            
        # plot specific time-series examples
        df_true = -0.25
        FDWN_time_series1 = simulate_FDWN(signal_length, df_true)
        FDWN_time_series2 = simulate_FDWN(signal_length, df_true)
        intFDWN_time_series1 = FGN_to_FBM(FDWN_time_series1) # integrate
        intFDWN_time_series2 = FGN_to_FBM(FDWN_time_series2) # integrate
        
        fig1, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(FDWN_time_series1, color='blue', label = 'realization 1')
        ax1.plot(FDWN_time_series2, color='red', label = 'realization 2')
        ax1.grid()
        title1 = 'FDWN examples, d = {:.2}'.format(df_true)
        ax1.set_title(title1)
        ax1.set_xlabel('time index')
        ax1.set_ylabel('amplitude')
        ax1.legend()
        
        ax2.plot(intFDWN_time_series1, color='blue', label = 'realization 1')
        ax2.plot(intFDWN_time_series2, color='red', label = 'realization 2')
        ax2.grid()
        title2 = 'integrated FDWN, d = {:.2}, compare to FBM with H = {:.2}'.format(df_true+1,df_true+.5)
        ax2.set_title(title2)
        ax2.legend()
        ax2.set_xlabel('time index')
        ax2.set_ylabel('amplitude')        
        plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)            
           
if __name__ == "__main__":
    main()