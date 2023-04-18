import os
import json
import h5py
import joblib
import gan_train
import numpy as np
import pandas as pd
import data_loading
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.noise_evaluation as evalnoise


class MyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle numpy objects and arrays
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if np.iscomplexobj(obj):
            return abs(obj)
        else:
            return super(MyEncoder, self).default(obj)


def plot_losses(train_hist_df, save, output_path):
    """
    Plot the losses of D and G recorded in the train_hist dictionary during training
    :param train_hist_df: Dataframe containing batch-level training metrics
    :param model_name: String Name of Model
    :param save: Boolean whether to save results_plots, or plot to console
    :param output_path: Path to save results_plots to
    :return: None
    """
    D_loss, G_loss = train_hist_df["Loss_D"], train_hist_df["Loss_G"]
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(f"GAN Training Loss")
    plt.plot(range(len(D_loss)), D_loss, color="red", alpha=0.4, label="Discriminator Loss")
    plt.plot(range(len(G_loss)), G_loss, color="blue", alpha=0.4, label="Generator Loss")
    D_loss_ma = D_loss.rolling(window=int(len(D_loss) / 100)).mean()
    G_loss_ma = G_loss.rolling(window=int(len(G_loss) / 100)).mean()
    plt.plot(range(len(D_loss_ma)), D_loss_ma, color="red")
    plt.plot(range(len(G_loss_ma)), G_loss_ma, color="blue")
    plt.ylabel("Loss")
    plt.xlabel("Batch number")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(output_path, "Gan_training_loss.png"), dpi=200)
    plt.close('all')

    D_x, D_G_z = train_hist_df["D(x)"], train_hist_df["D(G(z2))"]
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(f"Critic Output")
    plt.plot(range(len(D_x)), D_x, color="red", alpha=0.4, label="D(x)")
    plt.plot(range(len(D_G_z)), D_G_z, color="blue", alpha=0.4, label="D(G(z))")
    D_x_ma = D_x.rolling(window=int(len(D_x) / 100)).mean()
    D_G_z_ma = D_G_z.rolling(window=int(len(D_G_z) / 100)).mean()
    plt.plot(range(len(D_x_ma)), D_x_ma, color="red")
    plt.plot(range(len(D_G_z_ma)), D_G_z_ma, color="blue")
    if "GP" in train_hist_df.columns:
        GP = train_hist_df["GP"]
        plt.plot(range(len(GP)), GP, color="green", alpha=0.4, label="GP")
        GP_ma = GP.rolling(window=int(len(GP) / 100)).mean()
        plt.plot(range(len(GP_ma)), GP_ma, color="green")
    plt.ylabel("Discriminator Output")
    plt.xlabel("Batch number")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(output_path, "Gan_D_output.png"), dpi=200)
    plt.close('all')


def plot_waveforms(data, num_waveforms, dataset_name, output_path, save):
    """
    :param data:
    :param num_waveforms:
    :param dataset_name:
    :param waveform_type:
    :param output_path: Path to save directory
    :param save:
    :return:
    """
    if not os.path.isdir(f"{output_path}/waveform_plots/"):
        os.makedirs(f"{output_path}/waveform_plots/")
    for i in range(num_waveforms):
        waveform = data[i, :]
        print(f"length of waveform: {len(waveform)}")
        if len(waveform) > 10000:
            waveform = waveform[:10000]
        if waveform.dtype == np.complex128 or waveform.dtype == np.complex64:
            plt.plot(range(len(waveform)), np.real(waveform), alpha=0.7, label="I Component", color="blue")
            plt.plot(range(len(waveform)), np.imag(waveform), alpha=0.7, label="Q Component", color="green")
            plt.legend()
        else:
            waveform = data[i, :]
            plt.plot(range(len(waveform)), waveform, alpha=0.8, color="blue")
        plt.xlabel("Time Index", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.grid()
        if save:
            plt.savefig(os.path.join(output_path, f"waveform_plots/{dataset_name}_example_{i}.png"), dpi=200)
        plt.close('all')


def plot_spectrograms(data, num_spectrograms, dataset_name, spectrogram_type, output_path, save):
    """
    Plot example log-magnitude spectrograms
    :param data: example spectrograms
    :param num_spectrograms: number to plot
    :param dataset_name: Name for dataset source (Gen/Targ)
    :param spectrogram_type: Dataset representation string
    :param output_path: Path to save directory
    :param save: Save figures to the save directory
    :return:
    """
    wavetype_filename = spectrogram_type.replace(" ", "_")
    spectrogram_dir = os.path.join(output_path, 'spectrogram_plots')
    if not os.path.isdir(spectrogram_dir):
        os.makedirs(spectrogram_dir)
    #if data.shape[1] == 2:
    #    data = data_loading.pack_to_complex(data)
    for i in range(num_spectrograms):
        spectrogram = data[i, 0]
        #spectrogram = np.log10(np.abs(spectrogram) ** 2)
        print(f"spectrogram shape: {spectrogram.shape}")
        plt.imshow(spectrogram)
        plt.colorbar()
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        if save:
            plt.savefig(os.path.join(spectrogram_dir, f'{dataset_name}_example_{i}.png'), dpi=500)
        plt.close('all')


def waveform_comparison(gen_data, targ_data, output_path, common_yscale=True):
    """
    Plot multiple target and generated waveforms for visual comparison
    :param gen_data: Generated waveform dataset
    :param targ_data: Target waveform dataset
    :param output_path: Path to save directory
    :param common_yscale: Use a common y-axis range across subplots
    :return:
    """
    num_waveforms = 3
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=common_yscale)

    min_val = min(np.amin(targ_data[:3, :]), np.amin(gen_data[:3, :]))
    max_val = min(np.amin(targ_data[:3, :]), np.amin(gen_data[:3, :]))

    for i in range(num_waveforms):
        gen_waveform = gen_data[i, :]
        targ_waveform = targ_data[i, :]
        axs[i, 1].plot(range(len(gen_waveform)), gen_waveform, alpha=1, linewidth=1, color="green")
        axs[i, 0].plot(range(len(targ_waveform)), targ_waveform, alpha=1, linewidth=1, color="blue")
        axs[i, 1].grid()
        axs[i, 0].grid()
        axs[i, 0].margins(x=0, y=0.05)
        axs[i, 1].margins(x=0, y=0.05)
        if common_yscale:
            axs[i, 0].set_ylim((min_val))
            axs[i, 0].set_ylim((max_val))
            axs[i, 1].yaxis.set_ticks_position('none')
            if i != 2:
                axs[i, 1].xaxis.set_ticks_position('none')
                axs[i, 0].xaxis.set_ticks_position('none')

        axs[i, 0].set_ylabel("Amplitude", rotation=90)
    axs[0, 1].set_title("Generated")
    axs[0, 0].set_title("Target")
    axs[2, 1].set_xlabel("Time Index")
    axs[2, 0].set_xlabel("Time Index")
    plt.tight_layout()
    if common_yscale:
        plt.subplots_adjust(hspace=0.1, wspace=0.05)
    else:
        plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(output_path, "waveform_comp.png"), dpi=300)
    plt.show()
    plt.close('all')


def load_generated(G, n_samples, class_labels, dataset, batch_size, device):
    """
    create generated distribution, sampled from Generator for GAN evaluation
    :param batch_size: number of samples to generate per iteration
    :param G: Generator model object
    :param n_samples: Number of samples to be generated
    :param dataset: Dataset Name
    :param device: GPU device ID number
    :return: Generated data
    """
    G.eval()
    gen_data = []
    num_generated_samples = 0
    print(f"Generating {n_samples} fake samples: ")
    i = 0
    while num_generated_samples < n_samples:
        remaining_samples = n_samples - num_generated_samples
        curr_num_samples = batch_size if remaining_samples >= batch_size else remaining_samples
        z = data_loading.get_latent_vectors(curr_num_samples, 100, False, device)
        fake = G(z)
        fake = fake.detach().cpu()
        gen_data.append(fake)
        num_generated_samples += len(fake)
        i += 1
    gen_data = torch.cat(gen_data, dim=0).numpy()
    gen_data = gen_data[:n_samples] if len(gen_data) > n_samples else gen_data
    return gen_data


def load_target(dataset, d_type, dist_name):
    """
    Load target distribution from h5 file and process it into the proper format
    :param dataset: Dataset name
    :param d_type: Data-type (Complex/float)
    :param dist_name: Distribution name (Test/validation)
    :return: Target distribution, and supporting info
    """
    h5f = h5py.File(f"./Datasets/{dataset}/{dist_name}.h5", 'r')
    targ_dataset = h5f['train'][:]
    h5f.close()
    targ_data = np.array(targ_dataset[:, 1:]).astype(d_type)
    targ_labels = np.array(np.real(targ_dataset[:, 0])).astype(int)
    n_samples = len(targ_data)
    return targ_data, targ_labels, n_samples


def load_test_distributions(train_specs_dict, G, transformer, device, output_path):
    """
    Load in Generated and Target test distributions used for evaluation of GAN performance
    :param train_specs_dict: GAN configuration dictionary
    :param G: Generator model
    :param transformer: data scaler-transformer model
    :param device: GPU device ID number
    :param output_path: Path to save directory
    :return: Test distributions (Target/Generated)
    """
    dataset = train_specs_dict["dataloader_specs"]["dataset_specs"]["data_set"]
    d_type = float  # assuming real-valued target data
    targ_data, targ_labels, n_samples = load_target(dataset, d_type, "test")

    if os.path.exists(f"{output_path}/gen_distribution.h5"):
        print("loading presaved gen-distribution")
        h5f = h5py.File(f"{output_path}/gen_distribution.h5", 'r')
        gen_data = h5f['test'][:]
        h5f.close()
    else:
        print("Creating new gen-distribution")
        pad_length = train_specs_dict['dataloader_specs']['dataset_specs']["pad_length"]
        transform = train_specs_dict['dataloader_specs']['dataset_specs']["transform_type"]
        nperseg = train_specs_dict['dataloader_specs']['dataset_specs']["nperseg"]
        noverlap = train_specs_dict['dataloader_specs']['dataset_specs']["noverlap"]
        fft_shift = train_specs_dict["dataloader_specs"]["dataset_specs"]["fft_shift"]
        data_rep = train_specs_dict['dataloader_specs']['dataset_specs']["data_rep"]
        batch_size = train_specs_dict['dataloader_specs']['batch_size']
        quantize = train_specs_dict['dataloader_specs']['dataset_specs']["quantize"]
        scale_data = train_specs_dict['dataloader_specs']['dataset_specs']["data_scaler"]

        class_labels_tensor = torch.tensor(targ_labels, dtype=torch.int).to(device)
        gen_data = load_generated(G, n_samples, class_labels_tensor, dataset, batch_size, device)

        if (scale_data is not None) and (quantize is None):
            if transformer is not None:
                print("Inverse feature-based scaling")
                gen_data_shape = gen_data.shape
                gen_data = gen_data.reshape(gen_data_shape[0], -1)
                gen_data = transformer.inverse_transform(gen_data)
                gen_data = gen_data.reshape(gen_data_shape)
            elif quantize is None:
                print("Inverse global Min-Max scaling")
                dims = (0, 2) if len(gen_data.shape) == 3 else (0, 2, 3)
                cmin, cmax = np.amin(gen_data, axis=dims),  np.amax(gen_data, axis=dims)
                if len(gen_data.shape) == 3:
                    cmin = cmin[np.newaxis, :, np.newaxis]
                    cmax = cmax[np.newaxis, :, np.newaxis]
                else:
                    cmin = cmin[np.newaxis, :, np.newaxis, np.newaxis]
                    cmax = cmax[np.newaxis, :, np.newaxis, np.newaxis]

                feature_max, feature_min = 1, -1
                gen_data = (gen_data - feature_min) / (feature_max - feature_min)
                gen_data = gen_data * (cmax - cmin) + cmin

        if quantize is not None:
            quant_transformer = joblib.load(os.path.join(output_path, 'quantize_transformers.gz'))
            gen_data = data_loading.inverse_quantile_transform(gen_data, quant_transformer, type=quantize)

        if transform is not None:
            plot_spectrograms(gen_data, 5, "Gen", "", output_path, True)
            if data_rep != "IQ":
                gen_data = data_loading.phase_magnitude_to_iq(gen_data, data_rep)
            gen_data = data_loading.pack_to_complex(gen_data)
            if fft_shift:
                gen_data = np.fft.ifftshift(gen_data, axes=(1,))
            oneside = True   # assuming real-valued target data
            gen_data = data_loading.frequency_to_waveform(gen_data, type="stft", fs=2, nperseg=nperseg, noverlap=noverlap, onesided=oneside)
            if oneside:
                gen_data = np.real(gen_data)
        else:
            if len(gen_data.shape) >= 3 and gen_data.shape[1] == 1:
                gen_data = gen_data.squeeze(axis=1)
            elif len(gen_data.shape) >= 3 and gen_data.shape[1] == 2:
                gen_data = data_loading.pack_to_complex(gen_data)
        if pad_length is not None and pad_length > 0:
            gen_data = data_loading.unpad_signal(gen_data, pad_length)

        h5f = h5py.File(os.path.join(output_path, 'gen_distribution.h5'), "w")
        h5f.create_dataset('test', data=gen_data)
        h5f.close()

    assert gen_data.shape == targ_data.shape, f"Generated and Target test distributions are not the same: " \
                                              f"Gen {gen_data.shape} =/= Targ {targ_data.shape}"
    return targ_data, gen_data


def test_gan(G_net, train_hist_df, output_path, device, specs):
    """
    Evaluate performance of Generator and save performance metrics results_plots/ tables
    :param G_net: Generator model
    :param train_hist_df: Dataframe of batch-level training KPIs
    :param output_path: Path to save directory
    :param device: GPU device ID number
    :param specs: GAN configuraion dictionary
    :return: None
    """
    plot_losses(train_hist_df, specs["save_model"], output_path)
    if os.path.exists(os.path.join(output_path,'distance_metrics.json')):
        with open(rf'{output_path}/distance_metrics.json', 'r') as F:
            metric_dict = json.load(F)
    else:
        metric_dict = {}
        metric_dict["config"] = output_path
    dataset = specs["dataloader_specs"]["dataset_specs"]["data_set"]
    data_scaler = joblib.load(os.path.join(output_path,'target_data_scaler.gz'))
    print(f'test_gan output path = {output_path}')
    targ_data, gen_data = load_test_distributions(specs, G_net, data_scaler, device, output_path)

    plot_waveforms(gen_data, 5, "gen", output_path, True)
    plot_waveforms(targ_data, 5, "targ", output_path, True)

    print("Noise Evaluation: ")
    with open(os.path.join(f'./Datasets/{dataset}','noise_params.json')) as F:
        noise_dict = json.load(F)
    noise_type = noise_dict['noise_type']
    param_distrib = noise_dict['param_distrib']
    param_value = noise_dict['param_value']
    common_yscale = False if noise_type == 'FBM' or noise_type == 'SAS' else True
    waveform_comparison(gen_data, targ_data, output_path, common_yscale)

    if noise_type == 'FGN' or noise_type == 'FBM' or noise_type == "FDWN":
        targ_hursts, gen_hursts, hurst_wasserstein = evalnoise.evaluate_fn(targ_data, gen_data, noise_type, output_path)
        metric_dict["targ_hursts"] = np.mean(targ_hursts)
        metric_dict["gen_hursts"] = np.mean(gen_hursts)
        metric_dict["hurst_wasserstein"] = hurst_wasserstein
    if noise_type == "shot":
        pulse_type = noise_dict["pulse_type"]
        amp_distrib = noise_dict["amp_distrib"]
        gen_median_event_rate, targ_median_event_rate, event_rate_wasserstein = \
            evalnoise.evaluate_sn(targ_data, gen_data, pulse_type, amp_distrib, param_value, output_path)
        metric_dict["gen_median_event_rate"] = gen_median_event_rate
        metric_dict["targ_median_event_rate"] = targ_median_event_rate
        metric_dict["event_rate_wasserstein"] = event_rate_wasserstein

    if noise_type == "SAS":
        gen_median_alpha, targ_median_alpha, alpha_dist = evalnoise.evaluate_sas_noise(targ_data, gen_data, param_value, output_path)
        metric_dict["gen_median_alpha"] = gen_median_alpha
        metric_dict["targ_median_alpha"] = targ_median_alpha
        metric_dict["alpha_dist"] = alpha_dist

    if noise_type == "BG":
        print("Evaluate BG:")
        targ_prob_median, gen_prob_median, targ_amp_ratio_median, gen_amp_ratio_median, \
        impulse_prob_dist, amp_ratio_dist = evalnoise.evaluate_bgn(targ_data, gen_data, param_value, output_path)
        metric_dict["targ_prob_median"] = targ_prob_median
        metric_dict["gen_prob_median"] = gen_prob_median
        metric_dict["targ_amp_ratio_median"] = targ_amp_ratio_median
        metric_dict["gen_amp_ratio_median"] = gen_amp_ratio_median
        metric_dict["impulse_prob_dist"] = impulse_prob_dist
        metric_dict["amp_ratio_dist"] = amp_ratio_dist
    psd_dist = evalnoise.eval_psd_distances(targ_data, gen_data, param_value, noise_type, output_path)
    metric_dict["geodesic_psd_dist"] = psd_dist
    with open(rf'{output_path}/distance_metrics.json', 'w') as F:
        F.write(json.dumps(metric_dict, cls=MyEncoder))


def retest_gan(dir_path):
    with open(os.path.join(dir_path, 'gan_train_config.json'), 'r') as fp:
        train_specs_dict = json.loads(fp.read())
    dataset = train_specs_dict["dataloader_specs"]["dataset_specs"]["data_set"]
    input_length = train_specs_dict["dataloader_specs"]['dataset_specs']["input_length"]
    train_specs_dict["checkpoint"] = os.path.join(dir_path,'checkpoint.tar')

    try:
        x = train_specs_dict["dataloader_specs"]["dataset_specs"]["quantize"]
    except KeyError:
        print("quantile transform = False")
        train_specs_dict["dataloader_specs"]["dataset_specs"]["quantize"] = False
        train_specs_dict['model_specs']['use_tanh'] = True
    G_net, _, _, _, _ = gan_train.init_GAN_model(train_specs_dict, input_length, dir_path, "cpu")
    train_hist_df = pd.read_csv(os.path.join(dir_path, 'gan_training_history.csv'))
    train_hist_df = pd.DataFrame(train_hist_df)
    print(f'dir path = {dir_path}')
    test_gan(G_net, train_hist_df, dir_path, "cpu", train_specs_dict)