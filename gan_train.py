import os
import time
import math
import json
import torch
import joblib
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import gan_evaluation
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from models import DirectWaveGAN, SFTFGAN
from data_loading import TargetDataset, get_latent_vectors
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def print_training_metrics(train_report_dict, num_epochs, num_batches):
    """
    Print string with all GAN related training metrics
    :param train_report_dict: GAN Configuration dictionary
    :param num_epochs: total number of training epochs
    :param num_batches: total number of batches
    :return:
    """
    if 'GP' in train_report_dict:
        loss_string = f"[{train_report_dict['epoch_num']}/{num_epochs}][{train_report_dict['batch_num']}/{num_batches}]" \
                      f"\tLoss_D: {train_report_dict['Loss_D']:.2f}\tLoss_G: {train_report_dict['Loss_G']:.2f}\t" \
                      f"D(x): {train_report_dict['D(x)']:.2f}\tD(G(z)): {train_report_dict['D(G(z1))']:.2f} " \
                      f"/ {train_report_dict['D(G(z2))']:.2f}\tGP: {train_report_dict['GP']:.2f}"
    else:
        loss_string = f"[{train_report_dict['epoch_num']}/{num_epochs}][{train_report_dict['batch_num']}/{num_batches}]" \
                      f"\tLoss_D: {train_report_dict['Loss_D']:.2f}\tLoss_G: {train_report_dict['Loss_G']:.2f}\t" \
                      f"D(x): {train_report_dict['D(x)']:.2f}\tD(G(z)): {train_report_dict['D(G(z1))']:.2f} " \
                      f"/ {train_report_dict['D(G(z2))']:.2f}"
    print(loss_string)


def time_since(t0):
    """
    Get time since t_0 in minutes and seconds
    :param t0: starting time of training in seconds
    :return: m: minutes
    :return s: seconds
    """
    now = time.time()
    s = now - t0
    m = math.floor(s / 60)
    s -= m * 60
    return m, s


def load_GAN_checkpoint(D_net, G_net, optimD, optimG, filepath, output_path, rank):
    """
    Load Check-pointed GAN model weights into untrained models and optimizers if a
    checkpoint path is specified in the config dictionary
    :param D_net: Discriminator Model
    :param G_net: Generator Model
    :param optimD: Discriminator Adam Optimizer
    :param optimG: Generator Adam Optimizer
    :param filepath: checkpoint filepath
    :param output_path: path to directory used for saving model run and evaluation
    :param rank: NVIDIA GPU device ID number
    :return: GAN model and optimizers with checkpoint model weights loaded in
    """
    if os.path.isfile(filepath):
        train_hist_df = pd.read_csv(output_path + "gan_training_history.csv", index_col=False)
        map_loc = f'cuda:{rank}' if rank != "cpu" else rank
        checkpoint = torch.load(filepath, map_location=f'{rank}')
        D_net.load_state_dict(checkpoint['D_state_dict'])

        G_state_dict = checkpoint['G_state_dict']
        for key in list(G_state_dict.keys()):
            G_state_dict[key.replace('.conv_layer.weight', '.0.weight')] = G_state_dict.pop(key)

        G_net.load_state_dict(G_state_dict)
        optimD.load_state_dict(checkpoint['optimD_state_dict'])
        optimG.load_state_dict(checkpoint['optimG_state_dict'])
        epoch = checkpoint['epoch'] + 1
    else:
        train_hist_df = pd.DataFrame([])
        epoch = 0
    return D_net, G_net, optimD, optimG, train_hist_df, epoch


def save_checkpoint(optimD, optimG, D_net, G_net, epoch, output_path, e=None):
    """
    Save GAN model and optimizer state dictionaries for checkpoints
    :param optimD: Discriminator Adam Optimizer
    :param optimG: Generator Adam Optimizer
    :param D_net: Discriminator Model
    :param G_net: Generator Model
    :param epoch: current training epoch
    :param output_path: path to results directory for model run
    :param e: current epoch used to specify if checkpoint is saved for epoch_checkpoints directory
    :return: None
    """
    if e is not None:
        output_path = f"{output_path}epoch_checkpoints/"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        file_path = os.path.join(output_path, f'checkpoint_epoch_{e}.tar')
    else:
        file_path = os.path.join(output_path, f'checkpoint.tar')
    if hasattr(D_net, "module") and hasattr(G_net, "module"):
        torch.save({'epoch': epoch, 'D_state_dict': D_net.module.state_dict(),
                    'G_state_dict': G_net.module.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
                    'optimG_state_dict': optimG.state_dict()}, file_path)
    else:
        torch.save({'epoch': epoch, 'D_state_dict': D_net.state_dict(),
                    'G_state_dict': G_net.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
                    'optimG_state_dict': optimG.state_dict()}, file_path)


def init_GAN_model(specs, input_length, output_path, rank):
    """
    Initialize GAN models and wrap them in DataParallel
    :param specs: training configuration dictionary
    :param input_length: data signal length
    :param output_path: Model save path
    :param rank: Nvidia GPU index
    :return G: Generator Model
    :return D: Discriminator Model
    :return optimG: Adam Optimizer for Generator
    :return optimD: Adam Optimizer for Discriminator
    :return specs: updated training configuration dictionary
    :return train_hist: Training history DataFrame
    :return e_0: initial epoch to train from checkpoint (default 0)
    """
    # 1D Convolutional GAN model trained on complex time-series waveforms
    if specs["dataloader_specs"]["dataset_specs"]["transform_type"] is None:
        model = DirectWaveGAN
        specs["model_specs"]["lower_resolution"] = input_length // specs["model_specs"]["scale_factor"] ** specs["model_specs"]["model_levels"]
    # 2D Convolutional GAN Model trained on spectrograms
    else:
        specs["model_specs"]["receptive_field_type"] = "standard"

        model = SFTFGAN
        specs["model_specs"]["lower_resolution"] = [input_dim // specs["model_specs"]["scale_factor"] **
                                                    specs["model_specs"]["model_levels"] for input_dim in input_length]
        specs["model_specs"]["input_dimension_parity"] = ["even" if input_dim % 2 == 0 else "odd" for input_dim in input_length]
    max_c, min_c = 1024, 32
    G_channel_list = [max_c // (2 ** i) if max_c // (2 ** i) > min_c else min_c for i in range(specs["model_specs"]["model_levels"])]
    G_channel_list = G_channel_list + [specs["model_specs"]["num_channels"]]
    D_channel_list = G_channel_list[::-1]
    specs["model_specs"]["G_channel_list"], specs["model_specs"]["D_channel_list"] = G_channel_list, D_channel_list
    if rank is None:
        rank = f"cuda:{rank}"
    print(f"Rank: {rank}")
    D = model.D(**specs["model_specs"]).to(rank)
    G = model.G(**specs["model_specs"]).to(rank)

    # Initialize Generator and Discriminator Optimizers
    optimD = torch.optim.Adam(D.parameters(), **specs["optim_params"]['D'])
    optimG = torch.optim.Adam(G.parameters(), **specs["optim_params"]['G'])
    # Load GAN model checkpoint if one is specified in the configuration dictionary
    e_0 = 1
    train_hist = []
    if specs["checkpoint"] is not None:
        D, G, optimD, optimG, train_df, e_0 = load_GAN_checkpoint(D, G, optimD, optimG, specs["checkpoint"],
                                                                  output_path, rank)
        train_hist = train_df.to_dict('records')
    return G, D, optimG, optimD, specs, train_hist, e_0


def gan_train(rank, specs=None, output_path=None):
    """
    GAN training method
    :param rank: Current GPU device ID number
    :param world_size: Number of available GPUs on the system
    :param specs: GAN configuration dictionary
    :param output_path: Path to directory used to save GAN and evaluation
    :return: None
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(int(rank))
    torch.manual_seed(np.random.randint(100000))
    specs["torch_seed"] = torch.random.initial_seed()
    # Initialize DataLoader and Data-scaler for inverse scaling of Generates on backend during evaluation
    dataset = TargetDataset(**specs["dataloader_specs"]['dataset_specs'])
    data_loader = DataLoader(dataset, batch_size=int(specs["dataloader_specs"]['batch_size']),
                             sampler=None, shuffle=True, pin_memory=False, num_workers=16)
    specs["dataloader_specs"]['dataset_specs']["input_length"] = dataset.input_length
    specs["dataloader_specs"]['dataset_specs']["pad_length"] = dataset.pad_length

    joblib.dump(dataset.transformer, f'{output_path}/target_data_scaler.gz')
    joblib.dump(dataset.quant_transformer, f'{output_path}/quantize_transformers.gz')

    if specs["dataloader_specs"]['dataset_specs']['data_scaler'] == "feature_standard":
        print("Turning off output tanh activation ")
        specs['model_specs']['use_tanh'] = False

    if specs['dataloader_specs']['dataset_specs']['quantize']:
        abs_max = torch.max(torch.abs(dataset.dataset)).item()
        print(f"Training abs-max: {abs_max}")
        specs['model_specs']["max_abs_limit"] = abs_max
        specs['model_specs']['use_tanh'] = True
        specs['model_specs']['quantized'] = True

    # Initialize new GAN model and optimizers (with option for loading from specified last checkpoint)
    G_net, D_net, optimG, optimD, specs, train_hist, start_epoch = init_GAN_model(specs, dataset.input_length,
                                                                                  output_path, rank)
    # Wrap D and G in Data Parallel module for faster training across all GPUs
    print("Single Process Multi-GPU Training (DataParallel)")
    gpu_list = [int(specs["start_gpu"]) + i for i in range(int(specs["num_gpus"]))]
    print(f"GPU list: {gpu_list}, rank: {rank}")
    G_net = DataParallel(G_net, output_device=rank, device_ids=gpu_list)
    D_net = DataParallel(D_net, output_device=rank, device_ids=gpu_list)
    D_net.train()
    G_net.train()

    if rank == int(specs["start_gpu"]):
        print(f"Generator: {G_net}")
        print(f"Discriminator: {D_net}")
    if specs["save_model"] and rank == int(specs["start_gpu"]):
        with open(output_path + 'gan_train_config.json', 'w') as fp:
            json.dump(specs, fp, cls=gan_evaluation.MyEncoder)
    start_time = time.time()
    num_batches = len(data_loader)
    report_rate = num_batches // 4

    # GAN epoch level training loop
    for e in range(start_epoch, specs["epochs"] + 1):
        epoch_time = time_since(start_time)
        data_loader_iter = iter(data_loader)
        if rank == specs["start_gpu"]:
            print(f"Epoch #{e}: {epoch_time[0]}m {int(epoch_time[1])}s")
        current_resolution = None

        # GAN batch level training loop
        for i in range(num_batches):
            report_vars = {"epoch_num": e, "batch_num": i}

            x_real, _ = next(data_loader_iter)

            G_net.zero_grad()  # clear the gradients before each generator round to prevent accumulating gradients
            D_net.zero_grad()  # clear the gradients before each discriminator round to prevent accumulating gradients

            for p in D_net.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # Update Discriminator model
            x_real = torch.tensor(x_real, device=device)
            z = get_latent_vectors(len(x_real), 100, specs["latent_type"], device)
            x_fake = G_net(z)

            if i == 0 and e == start_epoch:
                print(f"Real Batch Shape: {x_real.shape}")
                print(f"Gen Batch Shape: {x_gen.shape}")
            D_real = D_net(x_real)  # Predict validity of real data using discriminator.
            D_fake = D_net(x_gen.detach())  # Predict validity of generated data using discriminator.
            D_real = -D_real.mean()
            D_real.backward()
            D_fake = D_fake.mean()
            D_fake.backward()

            #  Compute Gradient Penalty for Discriminator Loss [Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py]
            # Gradient pentalty should be computed in scope of .backward() method call
            alpha = torch.rand(size=(x_real.shape[0], 1, 1), device=device) if len(x_real.shape) == 3 else \
                torch.rand(size=(x_real.shape[0], 1, 1, 1), device=device)
            alpha = alpha.expand(x_real.size())
            x_interpolated = alpha * x_real.data + (1 - alpha) * x_gen.data
            x_interpolated = Variable(x_interpolated, requires_grad=True)
            D_x_interpolated = D_net(x_interpolated)  # Predict validity of interpolated data using discriminator.
            gradients = grad(outputs=D_x_interpolated, inputs=x_interpolated,
                             grad_outputs=torch.ones(D_x_interpolated.size(), device=device),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.reshape(gradients.size(0), -1)
            # Compute MSE to make discriminator to be a 1-Lipschitz function
            gradient_penalty = 10.0 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty.backward()

            report_vars["Loss_D"] = D_fake.mean().item() - D_real.mean().item() + gradient_penalty.item()
            report_vars["GP"] = gradient_penalty.item()

            optimD.step()  # Update the discriminator.
            report_vars["D(x)"] = D_real.mean().item()
            report_vars["D(G(z1))"] = D_fake.mean().item()

            # Update Generator model
            if i % specs['D_updates'] == 0:
                for p in D_net.parameters():  # reset requires_grad
                    p.requires_grad = False  # they are set to False below in netG update
                D_fake = D_net(x_gen)
                g_loss = -torch.mean(D_fake)
                g_loss.backward()
                optimG.step()
            # Report KPIs for each training batch to console and to training log file
            report_vars["D(G(z2))"] = g_loss.item()
            report_vars["Loss_G"] = -1 * report_vars["D(G(z2))"]
            report_vars["time_elapsed"] = time_since(start_time)
            train_hist.append(report_vars)
            if i % report_rate == 0 and rank == specs["start_gpu"]:
                print_training_metrics(report_vars, specs['epochs'], len(data_loader))

        # Save GAN progress Checkpoints 50 times during training and overwrite current main checkpoint  every 50 epochs for restarts
        if e % 50 == 0 and e > 0 and e != specs['epochs'] - 1 and rank == specs["start_gpu"] and specs["save_model"]:
            print(f"{e}/{specs['epochs']}: Saving model checkpoint.")
            save_checkpoint(optimD, optimG, D_net, G_net, e, output_path)
            train_hist_df = pd.DataFrame(train_hist)
            train_hist_df.to_csv(output_path + "gan_training_history.csv", index=False)

    # Training finished: return GAN to evaluation method
    epoch_time = time_since(start_time)
    if rank == specs["start_gpu"]:
            print(f"Training_finished: {epoch_time[0]}m {int(epoch_time[1])}s")

    if specs["save_model"] and rank == specs["start_gpu"]:
        print("Beginning GAN Model Evaluation: ")
        train_hist_df = pd.DataFrame(train_hist)
        print("Saving final model checkpoint.")

        train_hist_df.to_csv(output_path + "gan_training_history.csv", index=False)
        save_checkpoint(optimD, optimG, D_net, G_net, specs["epochs"], output_path)

        if specs["eval_model"]:
            print("begin GAN generator evaluation")
            try:
                gan_evaluation.test_gan(G_net.eval(), train_hist_df, output_path, device, specs)
            except Exception as e:
                print(e)
