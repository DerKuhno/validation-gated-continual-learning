import logging

# helper function to init cl methods with correct parameters
def cl_method_params(config):

    # Default values:
    config.method_params.sample_size = config.batch_size # Number of samples from the replay buffer
    config.method_params.tmp_sample_size = config.batch_size # Number of samples from the new experience

    config.method_params.num_epochs_exp = int(100) # Original implementation is 1
    config.method_params.stopping_patience_exp = int(5) # Original implementation: no validation set
    config.method_params.lr_multiplier = float(1) # Original implementation is 1
    config.method_params.cl_max_steps_per_epoch = int(1000) # Like in initial training

    if config.cl_method == "agem":
        config.method_params.sample_size = 35 # Mimicing the replay of 10 Tasks

    elif config.cl_method == "gem":
        config.method_params.memory_strength = float(0.7) # Original implementation is 0.5 (Gamma γ in paper) 
        config.method_params.num_replay_batches = int(10) # Mimicing the replay of 10 Tasks
    
    elif config.cl_method == "mir":
        config.method_params.sample_size = int(700) # Mimicing the replay of 10 Tasks. Number of samples to choose worst from. Called subsample in Mir implementation of Avalanche
        config.method_params.batch_size_mem = 35 # Number of samples actually used

    elif config.cl_method == "er":
        config.method_params.sample_size = 280
        # Current Best Hyperparameters

    elif config.cl_method == "der":
        config.method_params.sample_size = 280
        config.method_params.alpha = float(0.1)
    
    elif config.cl_method == "ser":
        config.method_params.sample_size = 70 # Number of samples from the replay buffer
        config.method_params.alpha = float(0.3)
        config.method_params.beta = float(0.05)
    
    elif config.cl_method == "upper_bound":
        config.method_params.stopping_patience_exp = 20
        config.method_params.num_epochs_exp = 1000

    elif config.cl_method == "finetuning":
        pass # uses standard parameters

    elif config.cl_method == "lower_bound":
        config.method_params.stopping_patience_exp = 20
        config.method_params.num_epochs_exp = 1000

    else:
        raise ValueError("Please choose a correct cl_method or turn off CL")
    
    # Hyperparameters for the replay strategy
    config.method_params.buffer_size = 70_000 # Default value close to datapoints of one period

    if config.replay_strat == 'none':
        pass # nothing happens

    elif config.replay_strat == 'gss_greedy':
        config.method_params.buffer_mem_strength = 1
    
    elif config.replay_strat == 'latent_greedy':
        config.method_params.buffer_mem_strength = 2
    
    elif config.replay_strat == 'episode_balanced':
        pass # doesn't need any more parameters

    elif config.replay_strat == 'loss_greedy':
        config.method_params.prune_frac = 0.05 # Fraction of datapoints with highest loss to prune (likely to be outliers/close do anomalies)

    elif config.replay_strat == 'variance_greedy':
        config.method_params.var_samples = 7 # Number of forward passes to calculate the variance with dropout, must be > 1

    else:
        raise ValueError("Please choose a correct cl_method or turn off CL")

    # Hyperparameter search always overwrites the parameters!
    if config.hyperparameter_search:
        for key, item in config.hyperparameter_search.items():
            setattr(config.method_params, key, item)
            logging.info(f"Hyperparameters: \
              {key}: {item}")
            print(f"Hyperparameters: \
              {key}: {item}")
    
    for key, item in vars(config.method_params).items():
        logging.debug(f"Method param: {key}: {item}")
