#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for calculating fidelity and diversity metrics based on
normalized dynamic time warping distances.

Author: Adam Wunderlich
Date: June/July 2023
"""

import time
import os
import json
import h5py
import numpy as np
from dtaidistance import dtw
from multiprocessing import Process, Queue
from scipy import stats

def get_dtw_distance_matrices(model_path, targ_data, gen_data, window_size=32):
    '''
    Parameters
    ----------
    model_path : string
        Path to model run directory.
    targ_data : 2-D numpy array
        Test set of time series from target distribution.
    gen_data : 2-D numpy array
        Test set of time series from generated distribution.
    window_size : int, optional
        Window size for dynamic time warping (DTW) calculations.
        The default is 32.

    Returns
    -------
    dist_matrix_targ : 2-D numpy array
        Matrix of normalized DTW distances for target data.
    dist_matrix_targ_gen : 2-D numpy array
        Matrix of normalized DTW distances between target and generated data.

    '''
    fname = os.path.join(model_path, 'dtw_distances.h5')
    if os.path.isfile(fname):
        h5f = h5py.File(fname, 'r')
        dist_matrix_targ = np.array(h5f['dtw_targ'][:])
        dist_matrix_targ_gen = np.array(h5f['dtw_targ_gen'][:])
        window_size_old = h5f['window_size'][()]
        h5f.close()
    else:
        window_size_old = None

    if window_size != window_size_old:
        # dtw with the desired window size is not precomputed
        # need to calculate distance matrices
        print('Estimating DTW distances ...')
        sample_size = targ_data.shape[0]
        assert gen_data.shape[0] == sample_size
        t_max = np.amax(np.absolute(targ_data), axis=1)
        t_norm = targ_data/t_max[:,None]
        g_max = np.amax(np.absolute(gen_data), axis=1)
        g_norm = gen_data/g_max[:,None]

        # compute normalized DTW distances
        t_start = time.time()
        dtw_targ_gen = dtw.distance_matrix_fast(list(np.concatenate((t_norm, g_norm))), use_pruning = True, window = window_size)/window_size
        dist_matrix_targ = dtw_targ_gen[0:sample_size, 0:sample_size]
        dist_matrix_targ_gen = dtw_targ_gen[0:sample_size, sample_size:(2*sample_size)]
        t_end = time.time()
        print(f'DTW processing time = {(t_end - t_start):.2f} seconds')

        h5f = h5py.File(fname, "w")
        h5f.create_dataset('dtw_targ', data=dist_matrix_targ)
        h5f.create_dataset('dtw_targ_gen', data=dist_matrix_targ_gen)
        h5f.create_dataset('window_size', data=window_size)
        h5f.close()

    return dist_matrix_targ, dist_matrix_targ_gen

def compute_density_coverage_metrics(dist_matrix_targ, dist_matrix_targ_gen, k=10):
    '''
    Compute the density and coverage metrics defined in
    Naeem et al, "Reliable Fidelity and Diversity Metrics for Generative Models",
    Proc. International Conference on Machine Learning, 2020.


    Parameters
    ----------
    dist_matrix_targ : 2-D square numpy array
        Distance matrix for target test samples.
    dist_matrix_targ_gen : 2-D numpy array
        Distance matrix for target and generated test samples.
    k : int, optional
        Number of nearest neighbors to use for estimates. The default is 10.

    Returns
    -------
    density : float
        Density fidelity metric proposed in (Naeem et al., 2020).
    coverage : float
        Coverage diversity metric proposed in (Naeem et al., 2020).
    '''

    ind = np.argpartition(dist_matrix_targ, k+1, axis=-1)[..., :(k+1)]
    k_nearest = np.take_along_axis(dist_matrix_targ, ind, axis=-1)
    targ_nn_dists = k_nearest.max(axis=-1)
    tnn = targ_nn_dists[:,np.newaxis]
    density = (1 / k) * np.mean(np.sum((dist_matrix_targ_gen < tnn), axis=0))
    coverage = np.mean(np.amin(dist_matrix_targ_gen, axis=1) < targ_nn_dists)

    return density, coverage

def wilson_score_interval(phat, n, alpha=0.05):
    '''
    Wilson score confidence interval for a binomial proportion
    Reference: Agesti & Coull, Approximate Is Better than "Exact" for
    Interval Estimation of Binomial Proportions," The American Statistician, 1998

    Parameters
    ----------
    phat : float
        Estimate of binomial proportion.
    n : int
        Sample size.
    alpha : float, optional, default 0.05
        Significance level.

    Returns
    -------
    L : float
        Lower bound of 1-alpha confidence interval
    U : float
        Uower bound of 1-alpha confidence interval

    '''

    z = stats.norm.ppf(1-alpha/2)  # 1-alpha/2 quantile of standard normal distribution
    L = (phat + (z**2)/(2*n) - z*np.sqrt((phat*(1-phat) + (z**2)/(4*n))/n))/(1+(z**2)/n)
    U = (phat + (z**2)/(2*n) + z*np.sqrt((phat*(1-phat) + (z**2)/(4*n))/n))/(1+(z**2)/n)

    return L, U

def density_bootstrap_task(queue, dist_matrix_targ, dist_matrix_targ_gen, i, k=10):
    '''
    Compute single bootstrap density estimate.

    Parameters
    ----------
    queue : Queue object from multiprocessing module
        Used to pass results from each process.
    dist_matrix_targ : 2-D square numpy array
        Distance matrix for target test samples.
    dist_matrix_targ_gen : 2-D numpy array
        Distance matrix for target and generated test samples.
    i : int
        Bootstrap sample index.
    k : int, optional
        Number of nearest neighbors to use for estimate. The default is 10.

    Returns
    -------
    None.

    '''

    rng = np.random.default_rng() # initialize new instance of random number generator
    N = dist_matrix_targ.shape[0]
    targ_ind = np.arange(N)
    gen_ind = rng.choice(N, N)
    dist_matrix_targ_boot = np.zeros((N, N))
    dist_matrix_targ_gen_boot = np.zeros((N, N))
    for row in range(N):
        dist_matrix_targ_boot[row, :] = dist_matrix_targ[targ_ind[row], targ_ind]
        dist_matrix_targ_gen_boot[row, :] = dist_matrix_targ_gen[targ_ind[row], gen_ind]

    density, _ = compute_density_coverage_metrics(dist_matrix_targ_boot, dist_matrix_targ_gen_boot, k=10)
    queue.put((i, density))

def density_bootstrap_CI(dist_matrix_targ, dist_matrix_targ_gen, k=10,
                         num_bootstrap_samples= 1000, batch_size = 100, alpha=0.05):
    '''
    Multiprocessing parallel implementation of bootstrap confidence interval
    estimation for the density fidelity metric of (Naeem et al., 2020).

    Parameters
    ----------
    dist_matrix_targ : 2-D square numpy array
        Distance matrix for target test samples.
    dist_matrix_targ_gen : 2-D numpy array
        Distance matrix for target and generated test samples.
    k : int, optional
        Number of nearest neighbors to use for estimate. The default is 10.
    num_bootstrap_samples : int, optional
        Number of bootstrap samples. The default is 1000.
    batch_size : int, optional
        Batch size for parallel processing.
        Must divide evenly into num_bootstrap_samples. The default is 100.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    L : float
        Lower bound of 1-alpha bootstrap confidence interval for density metric.
    U : float
        Upper bound of 1-alpha bootstrap confidence interval for density metric.

    '''
    print('Estimating bootrap confidence interval for density metric')
    assert num_bootstrap_samples % batch_size == 0
    q = Queue()
    density_boot = np.zeros(num_bootstrap_samples)
    t_start = time.time()
    for i in range(0, num_bootstrap_samples, batch_size):
        # execute all tasks in a batch
        processes = [Process(target=density_bootstrap_task, args=(q, dist_matrix_targ, dist_matrix_targ_gen, j)) for j in range(i, i+batch_size)]
        # start all processes
        for process in processes:
            process.start()
        for process in processes:
            ret = q.get()
            density_boot[ret[0]] = ret[1]
        # wait for all processes to complete
        for process in processes:
            process.join()
    t_end = time.time()
    print(f'bootstrap processing time = {(t_end - t_start):.2f} seconds')
    # use percentile method to obtain bootstrap confidence interval
    L = np.quantile(density_boot, alpha/2)
    U = np.quantile(density_boot, 1-alpha/2)
    return L, U

def main():
    # tests

    target_path = '/data/noise-gan/Datasets/'
    gen_path = '/data/noise-gan/model_results/FBM/FBM_fixed_H50/2023-03-16_13-39-03'

    with open(os.path.join(gen_path, 'gan_train_config.json'), 'r') as fp:
        train_specs_dict = json.loads(fp.read())
    dataset = train_specs_dict["dataloader_specs"]["dataset_specs"]["data_set"]
    h5f = h5py.File(os.path.join(gen_path, 'gen_distribution.h5'), 'r')
    gen_data = h5f['test'][:]
    h5f.close()
    gen_data = np.array(gen_data, dtype=np.double)

    dist_name = 'test'
    h5f = h5py.File(os.path.join(target_path, f"{dataset}/{dist_name}.h5"), 'r')
    targ_dataset = h5f['train'][:]
    h5f.close()
    targ_data = np.array(targ_dataset[:, 1:], dtype=np.double)

    dtw_dist_matrix_targ, dtw_dist_matrix_targ_gen = get_dtw_distance_matrices(gen_path, targ_data, gen_data, window_size=32)
    dtw_density, dtw_coverage = compute_density_coverage_metrics(dtw_dist_matrix_targ, dtw_dist_matrix_targ_gen, k=10)

    # estimate bootstrap confidence interval for density
    # and wilson score interval for coverage
    # Note: Bootstrap for coverage metric results in bootstrap estimates
    # that are uniformly lower than original point estimate.
    # Similarly, botstrapping over target indices for density metric
    # results in bootstrap estimates that are uniformly lower than
    # original point estimate.

    sample_size = targ_data.shape[0]
    [dtw_coverage_L, dtw_coverage_U] = wilson_score_interval(dtw_coverage, sample_size, alpha=0.05)
    [dtw_density_L, dtw_density_U] = density_bootstrap_CI(dtw_dist_matrix_targ, dtw_dist_matrix_targ_gen,
                                                  num_bootstrap_samples= 10000,
                                                  batch_size = 200, alpha=0.05)

    print(f'DTW density = {dtw_density:.3f}, [{dtw_density_L:.3f}, {dtw_density_U:.3f}],')
    print(f'DTW coverage = {dtw_coverage:.3f}, [{dtw_coverage_L:.3f}, {dtw_coverage_U:.3f}]')

if __name__ == "__main__":
    main()





