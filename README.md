## <u>**Convolutional GANs for Classical Random Process Models**</u>

Pytorch implementation of **WaveGAN** and **STFT-GAN**, 1D and 2D convolutional GAN architectures trained on synthetic 
classical random process waveforms. Training is done using pytorch's Data_parallel which allows for single process multi-GPU training.
(Multi-process multi-GPU (DistributedDataParallel) is not compatible with the gradient penalty operation (autograd.grad) and is not 
recommended when using Wasserstein-GP loss).

The code allows for the automation of testing of many model configurations across different datasets, and allowing for running 
replicas of model/dataset configurations.


### <u>GAN Configuration Dictionary Attributes (training_specs_dict.py):</u>
* **save_model**: Save model checkpoint and training information to "./model_results" directory [Default: True]
* **checkpoint**: Train GAN starting from checkpoint path [Default: None]
* **epochs**: Number of GAN training epochs [Default: 500]
* **D_updates**: Number of Discriminator updates to every 1 Generator updates [Default: 1]
* **eval_model**: Evaluate final GAN model checkpoint [Default: True]
* **latent_type**: Type of latent random distribution [Default: uniform]
* **num_gpus**: Number of GPU devices used for training [Default: 8]
* **start_gpu**: First GPU device ID number [Default: 0]

* **model_specs**: GAN model attributes passed to model object
    * **model_levels**: Number of Discriminator and Generator convolutional layers [Default: 5]
    * **num_channels**: Number of waveform channels in target distribution [Default: 2]
    * **weight_init**: Weight initialization method [Default: kaiming_normal]
    * **scale_factor**: Convolutional stride factor [Default: 2]
    * **kernel_size**: Maximum 1D convolutional kernel length [Default: 128]
    * **receptive_field_type**: Define uniform kernel lengths across layers (standard) or progressively scaled kernels (kernel) [Default: kernel]
    * **gan_bias**: Use bias in the GAN convolutional layers [Default: False]
    * **latent_dim**: Number of dimensions in the latent space[Default: 100]
    * **wavegan**: Specify WaveGAN as the model used for training[Default: False]
    * **phase_shuffle**: Turn phase shuffle layers on [Default: True]
    * **gen_batch_norm**: Turn on generator BatchNormalization layer[Default: False]
    * **use_tanh**: Use tanh output activation in the generator [Default: True]

* **dataloader_specs**: Attributes passed to initialize Data_Loader object
    * **batch_size**: Number of samples per training batch [Default: 64]
    * **dataset_specs**: Dataset related attributes
       * **data_scaler**: Data-scaling setting [Default: global_min_max]
       * **data_set**: path to target distribution training dataset [Default: OFDM/allocation_sets/mod16_iq_random_cp_long_128fft_rblocks1_subs36_evm-25]
       * **num_samples**: number of samples in training set [Default: None]
       * **pad_signal**: length to zero-pad target distribution singals (used for STFTGAN and WaveGAN) [Default: None]
       * **stft**: Use STFTGAN and transform distribution to STFT representation [Default: False]
       * **nperseg**: STFT FFT window length [Default: 0]
       * **fft_shift**: Shift STFTs to have zero-frequency components [Default: False]
       * **transform_type**: Time-frequency transformation option [Default: stft]
       * **noverlap**: Window overlap percentage [Default: 0.5]
       * **data_rep**: Data representation used for target data [Default: IQ]
       * **quantize**: Use quantile transformation on target data [Default: None]

* **optim_params**: Pytorch Optimizer attributes
    * **D**:
       * **lr**: Discriminator learning rate [Default: 0.0001]
       * **betas**: Discriminator optimizer momentum beta settings [Default: (0.0, 0.9)]
    * **G**: 
       * **lr**: Generator learning rate [Default: 0.0001]
       * **betas**: Generator optimizer momentum beta settings [Default: (0.0, 0.9)]

### <u>Datasets:</u>

WaveGAN and STFT-GAN are trained on a large set of synthetically created noise time series datasets. Dataset folders contain a training 
set as well as a test set used separately for evaluation.

### <u>Training a model/configuration of models</u>

Training of our models would be done by running `!python main.py` in the terminal, after updating the GAN configuration dictionary 
(training_specs_dict.py) default values with any values of interest. Additionally, to run a set of model configurations
in an automated fashion, a configuration of tests can be run by passing a configuration table (csv file) to the main python module. 
Running `!python main.py --configs ./experiment_resources/test_configs.csv`, where the file contains columns labeled by desired keys in the GAN 
configuration dictionary that are to be changed across runs. Runs are saved in ./Model_Results/ with subdirectories named by their target dataset 
and other non-default configurations and and a time-stamp. Evaluation of the model is set to run at the termination of model training.
