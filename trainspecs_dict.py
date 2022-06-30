specs_dict = {'save_model': True,
              'eval_model': True,
              "latent_type": "uniform",  # "gaussian", "uniform
              "num_gpus": 4,
              "start_gpu": 0,
              "checkpoint": None,
              'D_updates': 1,  # 1, 5
              'epochs': 500,
              'model_specs': {
                  'latent_dim': 100,
                  'weight_init': 'kaiming_normal',  # "kaiming_normal", "orthogonal"
                  'scale_factor': 2,  # 2, 4
                  "kernel_size": 5,
                  "wavegan": False,
                  "phase_shuffle": False,
                  "receptive_field_type": "standard",  # "standard, kernel, dilation"
                  "gan_bias": False,
                  "gen_batch_norm": False,
                  "model_levels": 5,
                  'num_channels': 2,  # 1, 2
                  'use_tanh': True
              },
              'dataloader_specs': {
                    'dataset_specs': {
                        'data_scaler': None,  # "feature_standard"
                        "pad_signal": True,
                        'data_set': "FGN_fixed_H50",
                        'num_samples': 0,
                        'transform_type': "stft",
                        'nperseg': 64,
                        'noverlap': 0.5,  # 0.5, 0.75 0.875 0.9375 0.96875, 0.9921875
                        'fft_shift': False,
                        'data_rep': "IQ",  # "IQ",  "log_mag_IF, "log_mag_IQ"
                        'quantize': "channel"
                    },
                    'batch_size':  128},
              'optim_params': {
                    'D': {'lr': 0.0001, 'betas': (0.0, 0.9)},
                    'G': {'lr': 0.0001, 'betas': (0.0, 0.9)}}
              }
