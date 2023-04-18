import os
import h5py
import time
import torch
import warnings
import numpy as np
from scipy import signal
from sklearn import preprocessing
from scipy.stats import truncnorm
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer

os.environ['SSQ_GPU'] = '0'
warnings.filterwarnings(action='ignore')


def unpack_complex(iq_data):
    """
    Convert complex 2D matrix to 3D matrix with 2 channels for real and imaginary dimensions
    :param iq_data: numpy complex matrix (2D)
    :return: numpy floating point matrix (3D)
    """
    iq_real = iq_data.real
    iq_imaginary = iq_data.imag
    iq_real = np.expand_dims(iq_real, axis=1)    # Make dataset 3-dimensional to work with framework
    iq_imaginary = np.expand_dims(iq_imaginary, axis=1)    # Make dataset 3-dimensional to work with framework
    unpacked_data = np.concatenate((iq_real, iq_imaginary), 1)
    return unpacked_data


def pack_to_complex(iq_data):
    """
     convert 3D matrix with 2 channels for real and imaginary dimensions to 2D complex representation
    :param iq_data: numpy floating point matrix (3D)
    :return:  numpy complex matrix (2D)
    """
    num_dims = len(iq_data.shape)
    if num_dims == 2:
        complex_data = 1j * iq_data[:, 1] + iq_data[:, 0]
    elif num_dims == 3:
        complex_data = 1j * iq_data[:, 1, :] + iq_data[:, 0, :]
    else:
        complex_data = 1j * iq_data[:, 1, :, :] + iq_data[:, 0, :, :]
    return complex_data


def scale_dataset(data, data_set=None, scale_type=None):
    """
    Scale target distribution's range to [-1, 1] with multiple scaling options
    :param scale_type:
    :param data: Target distribution
    :param data_set: dataset name
    :return: scaled target distribution
    """
    # Feature Based data scaling:
    if scale_type == "feature_min_max":
        print("feature-based min_max scaling")
        data_shape = data.shape
        data = data.reshape(data_shape[0], -1)
        transformer = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        print("Performing Fit transform")
        data = transformer.fit_transform(data)
        data = data.reshape(data_shape)

    # Global Dataset scaling:
    elif scale_type == "global_min_max":
        print("channel-based global min_max scaling")
        transformer = None
        dims = (0, 2) if len(data.shape) == 3 else (0, 2, 3)
        cmin, cmax = np.amin(data, axis=dims),  np.amax(data, axis=dims)
        if len(data.shape) == 3:
            cmin = cmin[np.newaxis, :, np.newaxis]
            cmax = cmax[np.newaxis, :, np.newaxis]
        else:
            cmin = cmin[np.newaxis, :, np.newaxis, np.newaxis]
            cmax = cmax[np.newaxis, :, np.newaxis, np.newaxis]

        assert(len(cmin.shape) == len(data.shape))
        feature_max, feature_min = 1, -1
        data = (data - cmin) / (cmax - cmin)
        data = data * (feature_max - feature_min) + feature_min

    else:
        print("WARNING: No scaling applied.")
        transformer = None

    return data, transformer


class TargetDataset(Dataset):
    """
    Wrapper for dictionary dataset that can be easily loaded and used for training through PyTorch's framework.
    Pairs a training example with its label in the format (training example, label)
    """
    def __init__(self, data_set, data_scaler, pad_signal, transform_type="stft", nperseg=128,
                 noverlap=0.5, fft_shift=False, data_rep="IQ", quantize=None):

        """
        Load in target distribution, scale data to [-1, 1], and unpack any labels from the data
        :param fft_shift: Shift STFT to be zero-frequency centered
        :param nperseg: STFT FFT window length
        :param transform_type: Convert complex waveform to STFT
        :param pad_signal: Length of zero padding target distribution waveforms
        :param data_set: Name of dataset
        :param data_scaler: Name of scaling function option
        :return: PyTorch tensors
        """
        print(f"Loading in target distribution from ./Datasets/{data_set}/train.h5")
        start_time = time.time()
        h5f = h5py.File(rf"./Datasets/{data_set}/train.h5", 'r')
        dataset = h5f['train'][:]
        h5f.close()
        labels = np.real(dataset[:, 0]).astype(int)
        dataset = dataset[:, 1:]

        self.input_length = len(dataset[0, :])
        self.pad_length = 0
        if pad_signal and transform_type is None:
            dataset, self.pad_length = pad_signal_to_power_of_2(dataset)
            self.input_length = self.pad_length + self.input_length
        if transform_type is not None:
            dataset, self.pad_length = pad_signal_to_power_of_2(dataset)
            dataset = waveform_to_frequency(dataset, type=transform_type, fs=2, nperseg=nperseg, noverlap=noverlap)
            if fft_shift and transform_type == "stft":
                dataset = np.fft.fftshift(dataset, axes=(1,))
            self.input_length = (dataset.shape[1], dataset.shape[2])
        if dataset.dtype == complex: # Unpacking complex-representation to 2-channel representation
            dataset = unpack_complex(dataset).view(float)
        dataset = np.expand_dims(dataset, axis=1) if len(dataset.shape) < 4 else dataset
        if data_rep != "IQ":
            dataset = iq_to_phase_magnitude(dataset, data_rep)
        if quantize is not None:
            dataset, self.quant_transformer = quantile_transform(dataset, type=quantize)
            self.transformer = None
        else:
            dataset, self.transformer = scale_dataset(dataset, data_set, data_scaler)
            self.quant_transformer = None
        self.dataset = torch.from_numpy(dataset).float()
        self.labels = torch.from_numpy(labels).int()
        print(f"Processing time: {time.time() - start_time}s")

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


def get_latent_vectors(batch_size, latent_size, latent_type="gaussian", device="cuda:0"):
    """
    Load latent space variables and fake labels used for Generator
    :param latent_type: Uniform or Gaussian latent distribution
    :param batch_size: length of batch
    :param latent_size: lantent space random seed variable dimension
    :param device: nvidia-device object
    :return: latent variable pytorch-tensor and fake class labels
    """
    if latent_type == "gaussian":
        z = torch.randn(batch_size, latent_size, 1, device=device)
    elif latent_type == "uniform":
        z = torch.from_numpy(np.random.uniform(low=-1.0, high=1.0, size=(batch_size, latent_size, 1))).float().to(device)
    else:
        truncate = 1.0
        lower_trunc_val = -1 * truncate
        z = []  # assume no correlation between multivariate dimensions
        for dim in range(latent_size):
            z.append(truncnorm.rvs(lower_trunc_val, truncate, size=batch_size))
        z = np.transpose(z)
        z = torch.from_numpy(z).unsqueeze(2).float().to(device)
    return z


def iq_to_phase_magnitude(dataset, data_rep="log_mag_IF", nperseg=128):
    """
    Convert Complex IQ dataset representation to Instantaneous frequency-magnitude representation
    :param dataset: Complex waveform dataset
    :return: Instataneous Frequency-Magnitude Dataset
    """

    amplitude_channel = np.sqrt(dataset[:, 1] ** 2 + dataset[:, 0] ** 2)
    if "log" in data_rep:
        print("Log-Magnitude Transform")
        amplitude_channel = np.log10(amplitude_channel)
    amplitude_channel = np.expand_dims(amplitude_channel, axis=1)  # Make dataset 3-dimensional to work with framework

    if "IF" in data_rep:
        print("Instantaneous Frequency Transform")
        phase_channel = np.arctan2(dataset[:, 1], dataset[:, 0])
        phase_channel = np.unwrap(phase_channel)  # unwrapped_phase
        initial_phase = phase_channel[:, :, 0][:, :, np.newaxis]  # initial_phase
        # concatenate initial phase and instantaneous frequency
        phase_channel = np.concatenate((initial_phase, np.diff(phase_channel, axis=2)), axis=2)
        phase_channel = np.expand_dims(phase_channel, axis=1)    # Make dataset 3-dimensional to work with framework
        dataset = np.concatenate((amplitude_channel, phase_channel), 1)
    if "GD" in data_rep:
        print("Group Delay Transform: ")
        phase = np.arctan2(dataset[:, 1], dataset[:, 0])
        unwrapped_phase = np.unwrap(phase)  # unwrapped_phase
        initial_group = unwrapped_phase[:, 0, :][:, np.newaxis, :]  # initial_phase
        group_delay_channel = np.concatenate((initial_group, np.diff(unwrapped_phase, axis=1)), axis=1)
        dataset = np.concatenate((amplitude_channel, np.expand_dims(group_delay_channel, axis=1)), 1)
    elif "IQ" in data_rep:
        print("IQ Transform")
        dataset = np.concatenate((amplitude_channel, dataset), 1)
    return dataset


def phase_magnitude_to_iq(dataset, data_rep="log_mag_IF", nperseg=128, noverlap=0.5, signal_length=4096):
    """
    Convert Instantaneous-Frequency-Magnitude representation to IQ complex waveform dataset
    :param signal_length:
    :param noverlap:
    :param nperseg:
    :param data_rep:
    :param dataset: Instantaneous-Frequency-Magnitude dataset
    :return: Complex waveform dataset
    """
    print("Convert Magnitude and Phase to IQ")
    if "log" in data_rep:
        print("Log-Magnitude Inverse Transform")
        dataset[:, 0] = 10 ** dataset[:, 0]
    if "IF" in data_rep:
        print("Instantaneous Frequency Inverse Transform")
        dataset[:, 1] = np.cumsum(dataset[:, 1], axis=2)  # int_frequency_channel
    if "GD" in data_rep:
        print("Group Delay Inverse Transform")
        dataset[:, 1] = np.cumsum(dataset[:, 1], axis=1)  # int_frequency_channel
    elif "IQ" in data_rep:
        print("IQ Transform")
        dataset[:, 1] = np.arctan2(dataset[:, 2], dataset[:, 1])
    iq_data = dataset[:, 0] * np.exp(dataset[:, 1] * 1j)
    Q_channel = np.expand_dims(np.real(iq_data), axis=1)  # Make dataset 3-dimensional to work with framework
    I_channel = np.expand_dims(np.imag(iq_data), axis=1)  # Make dataset 3-dimensional to work with framework
    dataset = np.concatenate((Q_channel, I_channel), 1)
    return dataset


def pad_signal_to_power_of_2(waveform_dataset):
    """
    Add zero padding to signal to nearest power of 2
    :param waveform_dataset: Target Distribution
    :return: zero-padded target distribution, zero-padding length
    """
    waveform_length = waveform_dataset.shape[-1]
    if waveform_length & (waveform_length - 1) == 0:
        return waveform_dataset, 0
    found = False
    test_int = waveform_length
    next_power_of_2 = None
    while found is False:
        if test_int & (test_int - 1) == 0:
            found = True
            next_power_of_2 = test_int
        else:
            test_int += 1
    pad_length = next_power_of_2 - waveform_length
    waveform_dataset = np.hstack((waveform_dataset[:, -pad_length::], waveform_dataset))
    return waveform_dataset, pad_length


def unpad_signal(waveform_dataset, pad_length):
    """
    Remove zero-padding of signal
    :param waveform_dataset: zero-padded dataset
    :param pad_length: length of zero-padding
    :return: waveform dataset
    """
    if pad_length > 0:
        if len(waveform_dataset.shape) == 3:
            waveform_dataset = waveform_dataset[:, :, pad_length:]
        else:
            waveform_dataset = waveform_dataset[:, pad_length:]
        return waveform_dataset
    else:
        return waveform_dataset


def waveform_to_frequency(dataset, type="stft", fs=2, nperseg=128, noverlap=0.5):
    print(f"Mapping timeseries dataset to {type}")
    onesided = False if dataset.dtype == complex else True
    signal_length = dataset.shape[1]
    num_samples = dataset.shape[0]
    print(f'signal_lengh = {signal_length}, num_samples = {num_samples}, nperseg = {nperseg}, noverlap = {noverlap}')
    time_resolution = int(signal_length / (nperseg * (1 - noverlap))) + 1
    freq_resolution = (nperseg // 2) + 1 if onesided else nperseg
    transform_dataset = np.zeros((num_samples, freq_resolution, time_resolution), dtype=complex)
    for i, x in enumerate(dataset):
        if len(dataset) > 10 and i % (len(dataset) // 10) == 0:
            print(i, end=", ")
            if i % (len(dataset) // 10) == 0:
                print("")
        if type == "stft":
            _, _, transform = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=int(nperseg * noverlap),
                                          return_onesided=onesided, boundary="even")
        transform_dataset[i, :, :] = transform
    return transform_dataset


def frequency_to_waveform(dataset, type="stft", fs=2, nperseg=128, noverlap=0.5, onesided=False):
    waveform_dataset = []
    print(f"Mapping {type} dataset to time-series:", end=" ")
    dataset_dtype = dataset.dtype
    time_resolution = dataset.shape[-1] - 1
    signal_length = int(nperseg * (1 - noverlap) * time_resolution)
    num_samples = len(dataset)
    waveform_dataset = np.zeros((num_samples, signal_length), dtype=dataset_dtype)
    for i, transform in enumerate(dataset):
        if i % 1000 == 0:
            print(f"{i}/{len(dataset)}")
        if type == "stft":
            _, x = signal.istft(transform, fs, nperseg=nperseg, noverlap=int(nperseg * noverlap),
                                input_onesided=onesided)
        waveform_dataset[i, :] = x
    return waveform_dataset


def quantile_transform(dataset, type="feature"):
    print("Beginning Quantile Transform fitting routine: ")
    datashape = dataset.shape
    if type == 'channel':        # Convert distribution into two separate 1d channels for quantile transformation:
        dataset_fit = np.hstack([dataset[:, i].reshape((-1, 1)) for i in range(datashape[1])])
    else:
        dataset_fit = dataset.reshape((datashape[0], -1))
    dataset_fit = np.sort(dataset_fit, kind='mergesort', axis=0)
    quant_transformer = QuantileTransformer(n_quantiles=1024, subsample=len(dataset_fit), output_distribution='normal')
    quant_transformer.fit(dataset_fit)
    del dataset_fit

    if type == 'feature':
        dataset = dataset.reshape((datashape[0], -1))
        dataset_transformed = quant_transformer.transform(dataset)
        dataset_transformed = dataset_transformed.reshape(datashape)
    else:
        dataset = np.hstack([dataset[:, i].reshape((-1, 1)) for i in range(datashape[1])])

        num_flattened_samples = datashape[2] * datashape[3] if len(datashape) == 4 else datashape[2]
        num_samples = len(dataset) // num_flattened_samples
        dataset_transformed = np.empty(datashape)
        print("Beginning Quantile Transform of Target distribution: ")
        for i in range(num_samples):
            if i % (num_samples // 50) == 0:
                print(f"{i}/{num_samples}")
            sample = dataset[i * num_flattened_samples: (i + 1) * num_flattened_samples]
            sample_transformed = quant_transformer.transform(sample)
            if len(datashape) == 4:
                sample_transformed = np.hstack([sample_transformed[:, i].reshape((-1, 1, datashape[2], datashape[3]))
                                                for i in range(datashape[1])])
            else:
                sample_transformed = np.hstack([sample_transformed[:, i].reshape((-1, 1, datashape[2]))
                                                for i in range(datashape[1])])
            dataset_transformed[i] = sample_transformed
    return dataset_transformed, quant_transformer


def inverse_quantile_transform(dataset, quant_transformer, type="feature"):
    print("Beginning Quantile Transform of Generated distribution: ")
    datashape = dataset.shape
    if type == "feature":
        dataset = dataset.reshape((datashape[0], -1))
        dataset_transformed = quant_transformer.inverse_transform(dataset)
        dataset_transformed = dataset_transformed.reshape((-1, datashape[1], datashape[2], datashape[3]))
    else:
        temp_dataset = np.hstack([dataset[:, i].reshape((-1, 1)) for i in range(datashape[1])])
        num_flattened_samples = datashape[2] * datashape[3] if len(datashape) == 4 else datashape[2]
        nsamples = len(temp_dataset) // num_flattened_samples
        dataset_transformed = np.empty(datashape)
        print("Beginning Quantile Transform of Generated distribution: ")
        for i in range(nsamples):
            if i % (nsamples // 10) == 0:
                print(f"{i}/{nsamples}")
            sample = temp_dataset[i * num_flattened_samples: (i + 1) * num_flattened_samples]
            inv_transformed = quant_transformer.inverse_transform(sample)
            # Convert distribution back to original shape
            if len(datashape) == 4:
                inv_transformed = np.hstack([inv_transformed[:, i].reshape((-1, 1, datashape[2], datashape[3]))
                                             for i in range(datashape[1])])
            else:
                inv_transformed = np.hstack([inv_transformed[:, i].reshape((-1, 1, datashape[2]))
                                             for i in range(datashape[1])])
            dataset_transformed[i] = inv_transformed
    return dataset_transformed
