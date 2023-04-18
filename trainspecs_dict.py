specs_dict = {'save_model': True,  # False for debugging
              'epochs': 500,
              'D_updates': 1,  # 1, 5
              'eval_model': True,
              "latent_type": "uniform",  # "gaussian", "uniform"
              "num_gpus": 1,
              "start_gpu": 0,
              'model_specs': {
                  "model_levels": 5,
                  'num_channels': 2,  # 1, 2
                  'weight_init': 'kaiming_normal',  # "kaiming_normal", "orthogonal"
                  'scale_factor': 2,  # 2, 4
                  "kernel_size": 5,
                  "wavegan": False,
                  "phase_shuffle": False,
                  "gan_bias": False,
                  'latent_dim': 100,
                  "gen_batch_norm": False,
                  'use_tanh': True
              },
              'dataloader_specs': {
                    'dataset_specs': {
                        'data_scaler': 'feature_min_max',   #'feature_min_max', 'global_min_max'
                        "pad_signal": True,
                        'data_set': "FBM/FBM_fixed_H50",
                        'transform_type': "stft",
                        'nperseg': 128,
                        'noverlap': 0.5,  # 0.5, 0.75
                        'fft_shift': False,
                        'data_rep': "IQ",  # "IQ",  "log_mag_IF, "log_mag_IQ"
                        'quantize': None # data quantile tranformation: None, channel, feature
                    },
                    'batch_size':  128},
              'optim_params': {
                    'D': {'lr': 0.0001, 'betas': (0.0, 0.9)},
                    'G': {'lr': 0.0001, 'betas': (0.0, 0.9)}}
              }
