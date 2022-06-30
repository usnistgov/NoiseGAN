#!/usr/bin/env python3
"""
Module for creating noise datasets.

Author: Adam Wunderlich
Date: June 2022
"""

import os
import json
import h5py
import numpy as np
import scipy.signal as signal
import utils.fractional_noise_utils as fn
import utils.shot_noise_utils as sn
import utils.bg_noise_utils as bg
import utils.alpha_stable_noise_utils as asn


def create_bp_noise_dataset(num_samples, signal_length, param_distrib, num_bands=8, band_index=None):
    """
    Create dataset of real-valued bandpass noise, where each class consists of noise with
    frequency support determined by a bandpass filter.
    :param num_samples: number of time-series in dataset
    :param signal_length: time-series length
    :param param_distrib: 'single' band or 'multiple' bands
    :param num_bands: number of bands to spread across full bandwidth
    :param band_index: band index for distrib_type='single'
    :return: dataset (array of dimension (num_samples, signal_length+1), where the first column contains class labels)
             scale_coeffs (dict of scale coefficients)
             sos_coeffs (dict of SOS IIR filter coefficients)
    """
    sampling_frequency = 2  # Nyquist freq is 1 (normalized digital frequency)
    filter_order = 40
    half_bandwidth = 1.0 / (2 * num_bands + (num_bands - 1))
    full_bandwidth = 2.0 * half_bandwidth
    sos_coeffs = {}
    # design band-pass filters
    for num in range(num_bands):
        if num == 0:
            cutoff_freqs = [full_bandwidth]
            filter_type = "lowpass"
        elif num == num_bands - 1:
            cutoff_freqs = [full_bandwidth + (half_bandwidth*num)
                            + (full_bandwidth*(num-1))]
            filter_type = "highpass"
        else:
            cutoff_freqs = [full_bandwidth + (half_bandwidth*num)
                            + (full_bandwidth*(num-1)), full_bandwidth
                            + (half_bandwidth*num) + (full_bandwidth*num)]
            filter_type = "bandpass"

        # design IIR filter
        sos = signal.iirfilter(N=filter_order, Wn=cutoff_freqs, btype=filter_type, ftype='butter',
                               fs=sampling_frequency, output='sos')
        sos_coeffs[num] = sos

    if param_distrib == 'single':
        class_labels = np.ones(num_samples) * band_index
    elif param_distrib == 'multiple':
        class_labels = np.random.randint(0, num_bands, num_samples)

    dataset = np.zeros((num_samples, signal_length))
    # filter white noise with respective bandpass filter
    for k, label in enumerate(class_labels):
        sos = sos_coeffs[label]
        white_noise = np.random.normal(0, 1, size=signal_length)
        filtered_white_noise = signal.sosfiltfilt(sos, white_noise)
        dataset[k, :] = filtered_white_noise
    class_labels = np.reshape(class_labels, (num_samples, 1))
    dataset = np.hstack((class_labels, dataset))
    scale_coeffs = {}
    channel_max = dataset[1:, :].max()
    channel_min = dataset[1:, :].min()
    scale_coeffs['max_0'] = channel_max
    scale_coeffs['min_0'] = channel_min
    return dataset, scale_coeffs, sos_coeffs


def create_sn_dataset(num_samples, signal_length, param_distrib, pulse_type, amp_distrib, event_rate=None):
    """
    Create generalized shot noise (filtered poisson process) dataset
    :param num_samples: number of time-series in dataset
    :param signal_length: time-series length
    :param param_distrib: parameter distribution string (options: 'fixed', 'uniform', 'multimodal')
    :param pulse_type: shape of shot noise pulses (options: 'one_sided_exponential', 'linear_exponential', 'gaussian')
    :param amp_distrib: pulse amplitude distribution (options: 'exponential', 'rayleigh')
    :param event_rate: events per unit time
    :return: Outputs: dataset (array of dimension (num_samples, signal_length+1), where
                      the first column contains parameter values)
             scale_coeffs (dict of scale coefficients)
    """
    tau_d = 1  # pulse duration
    beta = 1  # mean pulse amplitude
    theta = 0.1  # normalized time step = delta_t/tau_d

    if param_distrib == 'fixed':
        event_rates = np.ones(num_samples)*event_rate
    elif param_distrib == 'uniform':
        param_range = (0.25, 3.0)
        event_rates = np.random.uniform(param_range[0], param_range[1],
                                        size=num_samples)
    elif param_distrib == 'multimodal':
        num_classes = 5
        class_locs = np.linspace(0.5, 2.5, num_classes)
        event_rates = np.zeros(num_samples)
        class_nums = np.random.randint(0, num_classes, num_samples)
        for k, class_num in enumerate(class_nums):
            event_rates[k] = np.random.normal(loc=class_locs[class_num],
                                              scale=0.1, size=1)
            if event_rates[k] <= 0.01:
                event_rates[k] = 0.01
    dataset = np.zeros((num_samples, signal_length))
    for k, param in enumerate(event_rates):
        dataset[k, :] = sn.simulate_shot_noise(signal_length, pulse_type,
                                               amp_distrib, event_rates[k],
                                               tau_d, beta, theta)
    param_values = np.reshape(event_rates, (num_samples, 1))
    dataset = np.hstack((param_values, dataset))
    scale_coeffs = {}
    channel_max = dataset[1:, :].max()
    channel_min = dataset[1:, :].min()
    scale_coeffs['max_0'] = channel_max
    scale_coeffs['min_0'] = channel_min
    return dataset, scale_coeffs


def create_bg_dataset(num_samples, signal_length, param_distrib, impulse_prob=None, sig0=0.1, sig1=1):
    """
    Create Bernoulli-Gaussian noise (Gaussian mixture model) dataset
    :param num_samples: number of time-series in dataset
    :param signal_length: time-series length
    :param param_distrib: parameter distribution string (options: 'fixed', 'uniform', 'multimodal')
    :param sig0: standard deviation of background noise component
    :param sig1: standard deviation of impulse noise component
    :param impulse_prob: probability of impulse
    :return: Outputs: dataset (array of dimension (num_samples, signal_length+1), where
                      the first column contains parameter values)
             scale_coeffs (dict of scale coefficients)
    """

    if param_distrib == 'fixed':
        impulse_probs = np.ones(num_samples)*impulse_prob
    elif param_distrib == 'uniform':
        param_range = (0.01, 0.99)
        impulse_probs = np.random.uniform(param_range[0], param_range[1],
                                        size=num_samples)
    elif param_distrib == 'multimodal':
        num_classes = 5
        class_locs = np.linspace(0.1, 0.5, num_classes)
        impulse_probs = np.zeros(num_samples)
        class_nums = np.random.randint(0, num_classes, num_samples)
        for k, class_num in enumerate(class_nums):
            impulse_probs[k] = np.random.normal(loc=class_locs[class_num],
                                              scale=0.05, size=1)
            if impulse_probs[k] <= 0.01:
                impulse_probs[k] = 0.01
            if impulse_probs[k] >= 0.99:
                impulse_probs[k] = 0.99
    dataset = np.zeros((num_samples, signal_length))
    for k, p in enumerate(impulse_probs):
        dataset[k, :] = bg.simulate_bg_noise(signal_length, p, sig0, sig1)

    param_values = np.reshape(impulse_probs, (num_samples, 1))
    dataset = np.hstack((param_values, dataset))
    scale_coeffs = {}
    channel_max = dataset[1:, :].max()
    channel_min = dataset[1:, :].min()
    scale_coeffs['max_0'] = channel_max
    scale_coeffs['min_0'] = channel_min
    return dataset, scale_coeffs


def create_sas_dataset(num_samples, signal_length, param_distrib, alpha=None):
    """
    Create dataset of symmetric alpha-stable (SAS) noise
    :param num_samples: number of time-series in dataset
    :param signal_length: time-series length
    :param param_distrib: parameter distribution string
                          (options: 'fixed', 'uniform', 'multimodal')
    :param alpha: characteristic exponent
    :return: Outputs: dataset (array of dimension (num_samples, signal_length+1),
                      where the first column contains parameter values)
             scale_coeffs (dict of scale coefficients)
    """

    if param_distrib == 'fixed':
        alphas = np.ones(num_samples)*alpha
    elif param_distrib == 'uniform':
        param_range = (0.5, 1.5)
        alphas = np.random.uniform(param_range[0], param_range[1],
                                        size=num_samples)
    elif param_distrib == 'multimodal':
        num_classes = 5
        class_locs = np.linspace(0.5, 1.5, num_classes)
        alphas = np.zeros(num_samples)
        class_nums = np.random.randint(0, num_classes, num_samples)
        for k, class_num in enumerate(class_nums):
            alphas[k] = np.random.normal(loc=class_locs[class_num],
                                              scale=0.05, size=1)
            if alphas[k] < 0.01:
                alphas[k] = 0.01
            if alphas[k] > 1.99:
                alphas[k] = 1.99
    dataset = np.zeros((num_samples, signal_length))
    for k, a in enumerate(alphas):
        dataset[k, :] = asn.simulate_sas_noise(signal_length, a)

    param_values = np.reshape(alphas, (num_samples, 1))
    dataset = np.hstack((param_values, dataset))
    scale_coeffs = {}
    channel_max = dataset[1:, :].max()
    channel_min = dataset[1:, :].min()
    scale_coeffs['max_0'] = channel_max
    scale_coeffs['min_0'] = channel_min
    return dataset, scale_coeffs


def create_fn_dataset(num_samples, signal_length, noise_type, param_distrib, Hurst_index=None):
    """
    Create factional noise dataset
    :param num_samples: number of time-series in dataset
    :param signal_length: time-series length
    :param noise_type: 'FGN','FBM', or 'FDWN'
    :param param_distrib: parameter distribution string [options: 'fixed', 'uniform low', 'uniform high', 'mulitmodal low', 'multimodal high']
    :param Hurst_index: Hurst index for 'fixed' distribution option
    :return: array of dimension (num_samples, signal_length+1), where the first column contains parameter values), dict of scale coefficients
    """
    if 'fixed' in param_distrib:
        param_values = np.ones(num_samples) * Hurst_index
    if 'uniform' in param_distrib:
        if 'low' in param_distrib:
            param_range = (0.01, 0.49)
        elif 'high' in param_distrib:
            param_range = (0.5, 0.99)
        else:
            param_range = (0.01, 0.99)
        param_values = np.random.uniform(param_range[0], param_range[1], size=num_samples)
    if 'multimodal' in param_distrib:
        if 'low' in param_distrib:
            num_classes = 5
            class_locs = np.linspace(0.05, 0.45, num_classes)
        elif 'high' in param_distrib:
            num_classes = 5
            class_locs = np.linspace(0.55, 0.95, num_classes)
        else:
            num_classes = 10
            class_locs = np.linspace(0.05, 0.95, num_classes)
        param_values = np.zeros(num_samples)
        class_nums = np.random.randint(0, num_classes, num_samples)
        for k, class_num in enumerate(class_nums):
            param_values[k] = np.random.normal(loc=class_locs[class_num],
                                               scale=0.025, size=1)
            if param_values[k] <= 0.01:
                param_values[k] = 0.01
            elif param_values[k] >= 0.99:
                param_values[k] = 0.99
    if noise_type == 'FDWN':
        param_values = param_values - 0.5  # convert Hurst exponent to dfrac

    dataset = []
    for k, param in enumerate(param_values):
        if (noise_type == 'FGN') or (noise_type == 'FBM'):
            sample, _ = fn.simulate_FGN(signal_length, H=param)
            if noise_type == 'FBM':
                sample = fn.FGN_to_FBM(sample)
            dataset.append(sample)
        elif noise_type == 'FDWN':
            if k % 1000 == 0:
                print(f'computing waveform {k} of {num_samples-1}')
            sample = fn.simulate_FDWN(signal_length, dfrac=param)
            dataset.append(sample)
    dataset = np.array(dataset)

    param_values = np.reshape(param_values, (num_samples, 1))
    dataset = np.hstack((param_values, dataset))
    scale_coeffs = {}
    channel_max = dataset[1:, :].max()
    channel_min = dataset[1:, :].min()
    scale_coeffs['max_0'] = channel_max
    scale_coeffs['min_0'] = channel_min
    return dataset, scale_coeffs


def save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib,
                 param_value=None, num_bands=8, band_index=None,
                 pulse_type=None, amp_distrib=None):
    """
    :param data_dir:
    :param num_samples: number of time-series in training dataset
    :param signal_length: time-series length
    :param noise_type: Noise distribution type: 'bandpass', 'shot', 'BG',
                        'SAS', 'FGN','FBM', or 'FDWN'
    :param param_distrib: parameter distribution string
    :param param_value: Hurst index for fractional noise,
                        event rate for generalized shot noise,
                        impulse_prob for BG noise, alpha for SAS noise
    :param num_bands: number of bands for bandpass noise
    :param band_index: index of band for single band of BP noise
    :param pulse_type: pulse type for shot noise
    :param amp_distrib: amplitude distribution for shot noise
    :return:
    """
    if noise_type == 'bandpass':
        train_dataset, scale_coeffs, sos_coeffs = create_bp_noise_dataset(num_samples, signal_length, param_distrib, num_bands, band_index)
        test_dataset, _, _ = create_bp_noise_dataset(num_samples // 4, signal_length, param_distrib, num_bands, band_index)
        dir_path = data_dir + rf'{noise_type}_{num_bands}_bands'
        if param_distrib == 'single':
            dir_path = dir_path + '_band' + str(band_index)
        else:
            dir_path = dir_path + '_all'
    elif noise_type == 'shot':
        train_dataset, scale_coeffs = create_sn_dataset(num_samples, signal_length, param_distrib, pulse_type, amp_distrib, event_rate=param_value)
        test_dataset, _ = create_sn_dataset(num_samples // 4, signal_length, param_distrib, pulse_type, amp_distrib, event_rate=param_value)
        dir_path = data_dir + rf'{noise_type}_{pulse_type}_{amp_distrib}_{param_distrib}'
        if param_distrib == 'fixed':
            dir_path = dir_path + '_ER' + str(int(param_value*100))
    elif noise_type == 'BG':
        train_dataset, scale_coeffs = create_bg_dataset(num_samples, signal_length, param_distrib, impulse_prob=param_value)
        test_dataset, _ = create_bg_dataset(num_samples // 4, signal_length, param_distrib, impulse_prob=param_value)
        dir_path = data_dir + rf'{noise_type}_{param_distrib}'
        if param_distrib == 'fixed':
            dir_path = dir_path + '_IP' + str(int(param_value*100))
    elif noise_type == 'SAS':
        train_dataset, scale_coeffs = create_sas_dataset(num_samples, signal_length, param_distrib, alpha=param_value)
        test_dataset, _ = create_sas_dataset(num_samples // 4, signal_length, param_distrib, alpha=param_value)
        dir_path = data_dir + rf'{noise_type}_{param_distrib}'
        if param_distrib == 'fixed':
            dir_path = dir_path + '_alpha' + str(int(param_value*100))
    else:  # fractional noise type
        train_dataset, scale_coeffs = create_fn_dataset(num_samples, signal_length, noise_type, param_distrib, param_value)
        test_dataset, _ = create_fn_dataset(num_samples // 4, signal_length, noise_type, param_distrib, param_value)
        dir_path = data_dir + rf'{noise_type}_{param_distrib}'
        if param_distrib == 'fixed':
            dir_path = dir_path + '_H' + str(int(param_value*100))

    noise_dict = {'num_train_samples': num_samples, 'num_test_samples': num_samples // 4, 'signal_length': signal_length,
                  'noise_type': noise_type, 'param_distrib': param_distrib, 'param_value': param_value, 'num_bands': num_bands,
                  'band_index': band_index, 'pulse_type': pulse_type, 'amp_distrib': amp_distrib}

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    h5f = h5py.File(rf'{dir_path}/train.h5', 'w')
    h5f.create_dataset('train', data=train_dataset)
    h5f.close()
    h5f = h5py.File(rf'{dir_path}/test.h5', 'w')
    h5f.create_dataset('train', data=test_dataset)
    h5f.close()
    with open(rf'{dir_path}/scale_factors.json', 'w') as F:
        F.write(json.dumps(scale_coeffs))
    with open(rf'{dir_path}/noise_params.json', 'w') as F:
        F.write(json.dumps(noise_dict))
    if noise_type == 'bandpass':
        with open(rf'{dir_path}/sos_coeffs.json', 'w') as F:
            F.write(json.dumps(sos_coeffs, cls=MyEncoder))
    print(f'saving {dir_path}')


class MyEncoder(json.JSONEncoder):
    # flexible encoder for writing JSON files
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(MyEncoder, self).default(obj)


def main():
    # make datasets
    data_dir = r'./Datasets/'
    num_samples = 16384  # number of samples in training set. Test set has 1/4
    signal_length = 4096
    num_bands = 8  # number of bands for bandpass noise

    for noise_type in ['FGN']:  # 'FDWN', 'FGN', 'FBM', 'bandpass', 'shot', 'BG', 'SAS'
        if noise_type == 'bandpass':
            for param_distrib in ['single']:
                if param_distrib == 'single':
                    for num in range(num_bands):
                        print(f'saving  BP noise dataset {num} of {num_bands}')
                        band_index = num
                        save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib,
                                     num_bands=num_bands, band_index=band_index)
                else:  # multiple bands
                    save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib, num_bands=num_bands)
        elif noise_type == 'shot':
            for param_distrib in ['fixed']:  # 'uniform', 'multimodal',
                for pulse_type in ['one_sided_exponential', 'linear_exponential', 'gaussian']:
                    for amp_distrib in ['exponential']:  # 'rayleigh', 'standard_normal'
                        if param_distrib == 'fixed':
                            for count, param_value in enumerate(np.arange(0.25, 3.25, 0.25)):
                                print(f'saving  shot noise dataset {count+1} of 12')
                                save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib, param_value,
                                             pulse_type=pulse_type, amp_distrib=amp_distrib)
                        else:
                            save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib,
                                         pulse_type=pulse_type, amp_distrib=amp_distrib)
        elif noise_type == 'BG':
             for param_distrib in ['fixed']:  # 'uniform', 'multimodal'
                 if param_distrib == 'fixed':
                            for count, param_value in enumerate([.01, .05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                                print(f'saving  BG noise dataset {count+1} of 12')
                                save_dataset(data_dir, num_samples,
                                             signal_length, noise_type,
                                             param_distrib, param_value)
                 else:
                    save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib,
                                 pulse_type=pulse_type, amp_distrib=amp_distrib)
        elif noise_type == 'SAS':
             for param_distrib in ['fixed']:  # 'uniform', 'multimodal'
                 if param_distrib == 'fixed':
                            for count, param_value in enumerate(np.linspace(.5, 1.5, 11)):
                                print(f'saving  SAS noise dataset {count+1} of 11')
                                save_dataset(data_dir, num_samples,
                                             signal_length, noise_type,
                                             param_distrib, param_value)
                 else:
                    save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib)
        elif noise_type == 'FGN' or 'FDWN' or 'FBM':
            for param_distrib in ['fixed']:  # 'uniform_low', 'uniform_high',  'multimodal_low', 'multimodal_high',
                if param_distrib == 'fixed':
                    for count, param_value in enumerate([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]):
                        print(f'saving  {noise_type} noise dataset {count+1} of 11')
                        save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib, param_value)
                else:
                    save_dataset(data_dir, num_samples, signal_length, noise_type, param_distrib)


if __name__ == "__main__":
    main()
