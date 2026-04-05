

    using DataStructures

    NSGA2_searchspace = OrderedDict(
        "N" => [10, 500, 10],        
        "η_cr" => [1.0, 100.0, 1.0],      
        "p_cr" => [0.1, 1, 0.05],      
        "η_m" => [1.0, 100.0, 1.0],
        "p_m" => [0.05, 1.0, 0.05]        
    )
    
    #TODO: MOEAD_DE the question of the weights  
    NSGA3_searchspace = OrderedDict(
        "N" => [10, 500],
        "η_cr" => [1.0, 100.0],
        "p_cr" =>[0.1, 1],
        "η_m" => [1.0, 100.0], 
        "partitions" => [10, 500],
    )

    SMS_EMOA_searchspace = OrderedDict(
        "N" => [10, 500],
        "η_cr" => [1.0, 100.0],
        "p_cr" => [0.1, 1],
        "η_m" =>[1.0, 100.0],
        "p_m" => [0.05, 1.0],   
        "n_samples" => [1000, 10000],  ###Slows down the optimization process significantly
    )

    SPEA2_searchspace = OrderedDict(
        "N" => [10, 500, 10],
        "η_cr" => [1.0, 100.0, 1.0],
        "p_cr" => [0.1, 1.0, 0.05],
        "η_m" => [1.0, 100.0, 1.0],
        "p_m" => [0.05, 1.0, 0.05]   
        )

    MOEAD_DE_searchspace = OrderedDict( 
        "npartitions" => [10, 500], # will be capped by constant MAX_MOEAD_POPULATION
        "F" => [0.1, 1.0],
        "CR" => [0.1, 1.0],
        "η" => [5.0, 100.0],
        "p_m" => [0.005, 1.0],
        "δ" =>[0.1, 1.0], 
        "s1" => [0.0005, 0.1], # Irrelavant for unconstrained problems
        "s2" => [1.0, 100.0] # Irrelavant for unconstrained problems
    )

# grows exponentially, must the changed locally
    if @isdefined(optuna)
        optuna_sampler_dict = OrderedDict(
            "NSGAIISampler" => optuna.samplers.NSGAIISampler,
            "CmaEsSampler" => optuna.samplers.CmaEsSampler,
            "TPESampler" => optuna.samplers.TPESampler,
            "RandomSampler" => optuna.samplers.RandomSampler,
            "QMCSampler" => optuna.samplers.QMCSampler,
            "NSGAIIISampler" => optuna.samplers.NSGAIIISampler,
            "GPSampler" => optuna.samplers.GPSampler,
            )

        sampler_vector =  collect(keys(optuna_sampler_dict))
    end


