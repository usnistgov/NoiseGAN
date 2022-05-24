import os
import copy
import torch
import warnings
import argparse
import datetime
import pandas as pd
from gan_train import gan_train
from trainspecs_dict import specs_dict
from experiment_resources.gan_configurator import Automated_config_setup
warnings.filterwarnings("ignore")


def parse_configurations():
    """
    Parse options from terminal call
    Options:
        --configs: Specify path to csv containing configuration settings used to overwrite default GAN config settings
        --repeats: Specify number of times to repeat model training
    :return: configurations DataFrame, Number of repeats, Number of GPU devices, Main GPU device ID
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, help='CSV of configuration settings')
    parser.add_argument('--repeats', type=int, help='number of configuration repeats')
    args = parser.parse_args()
    configs, repeat_number = args.configs, args.repeats
    if configs is not None:
        configurations_df = pd.read_csv(configs)
        return configurations_df, repeat_number
    else:
        return None, repeat_number


def update_specs_params(specs_dict, num_gpus):
    """
    Update Training specs dictionary to avoid confilcts
    :param specs_dict: GAN Configuration dictionary
    :param num_gpus: Number of GPU devices being used
    :return: Updated GAN Configuration dictionary
    """
    dataset = specs_dict['dataloader_specs']['dataset_specs']["data_set"]
    if dataset.find("_archive") and specs_dict['dataloader_specs']['dataset_specs']["data_rep"] is None:
        specs_dict["model_specs"]['num_channels'] = 1
    if specs_dict["model_specs"]["wavegan"]:
        print("WaveGAN run: ")
        specs_dict['dataloader_specs']['dataset_specs']["transform_type"] = None
        specs_dict['dataloader_specs']['dataset_specs']["nperseg"] = None
        specs_dict['dataloader_specs']['dataset_specs']["noverlap"] = None
        specs_dict["model_specs"]['num_channels'] = 1
        specs_dict["model_specs"]["scale_factor"] = 4
        specs_dict["model_specs"]["kernel_size"] = 25
        specs_dict["model_specs"]["gan_bias"] = True
        specs_dict["dataloader_specs"]['dataset_specs']["pad_signal"] = True
        specs_dict["D_updates"] = 5
        specs_dict["model_specs"]["receptive_field_type"] = "standard"
        specs_dict['optim_params']['D']['betas'] = (0.5, 0.9)
        specs_dict['optim_params']['G']['betas'] = (0.5, 0.9)
        specs_dict["latent_type"] = "uniform"
        specs_dict['model_specs']["phase_shuffle"] = True

        if specs_dict["num_gpus"] > torch.cuda.device_count():
            specs_dict["num_gpus"] = torch.cuda.device_count()
    if specs_dict['dataloader_specs']['dataset_specs']["transform_type"] == "stft":
        if specs_dict['dataloader_specs']['dataset_specs']["data_rep"] == "log_mag_IQ":
            specs_dict["model_specs"]['num_channels'] = 3
        else:
            specs_dict["model_specs"]['num_channels'] = 2
    return specs_dict


def config_name(config):
    """
    Create GAN configuration name used in output-directory based config
    :param config: GAN configuration dictionary
    :return: GAN name string
    """
    config_string = ""
    for factor, value in config.items():
        if factor in ["conditional", "progressive", "label_smoothing",
                      "instance_noise" "gradient_penalty"] and value:
            config_string += factor + "_"
        if factor in ["data_scaler", "data_set"]:
            config_string += value + "_"
        if factor == "kernel_size":
            config_string += f"{str(value)}kernel_"
        if factor == "model_levels":
            config_string += f"{str(value)}layers_"
        if factor == "batch_size":
            config_string += factor + "-" + str(value) + "_"
    print(config_string[:-1])
    return config_string[:-1]


def run_config(specs_dict, world_size, rank=0, gan_name=None):
    """
    Update configuration dictionary and runs GAN model
    :param rank: Main GPU ID number
    :param specs_dict: GAM Configuration dictionary
    :param world_size: Number of available GPUs
    :param gan_name: Name of GAN configuration used for saving purposes
    :return: None
    """
    specs_dict_updated = update_specs_params(specs_dict, world_size)

    output_dir = "./model_results/"    # Define the results directory and make one if it doesnt already exist
    gan_name = specs_dict_updated["dataloader_specs"]["dataset_specs"]["data_set"] if gan_name is None else gan_name
    gan_name = gan_name.replace("/", "_")
    output_path = f"{output_dir}{gan_name}/{str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-')}/"
    if specs_dict_updated["checkpoint"] is not None:
        output_path = os.path.dirname(specs_dict_updated["checkpoint"]) + "/"
    if specs_dict_updated["save_model"] and rank == specs_dict_updated["start_gpu"]:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    gan_train(rank, specs_dict_updated, output_path)


if __name__ == '__main__':
    print("Main GAN Framework Run: ")
    configs_df, num_repeats = parse_configurations()
    num_repeats = 1 if num_repeats is None else num_repeats
    rank = 0
    for repeat in range(num_repeats):
        if configs_df is not None:
            print(f"Automated GAN Factor Screening: Repeat #{num_repeats}:")
            init_configs = Automated_config_setup()
            for ind in configs_df.index:
                config = configs_df.iloc[ind, :]
                config_str = config_name(config)

                # copy must be passed so changes dont accumulate to config dictionary
                specs_dict_updated = init_configs.map_params(config, copy.deepcopy(specs_dict))
                world_size = int(specs_dict_updated["num_gpus"])
                rank = int(specs_dict_updated["start_gpu"])
                if specs_dict_updated is None:
                    continue
                name_suffix = "_".join([str(val) for val in config.values])
                print(f"GAN Configuration: {config_str}, Repeat # {repeat}")
                run_config(specs_dict_updated, world_size, rank, config_str)
        else:
            print(f"Single GAN Configuration: Repeat # {repeat}")
            world_size = int(specs_dict["num_gpus"])
            rank = int(specs_dict["start_gpu"])
            run_config(specs_dict, world_size, rank)
