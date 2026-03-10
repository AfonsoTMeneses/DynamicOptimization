

    using DataStructures

    NSGA2_searchspace = OrderedDict(
        "N" => [10, 500, 10],        
        "η_cr" => [1.0, 100.0, 1.0],      
        "p_cr" => [0.1, 1, 0.05],      
        "η_m" => [1.0, 100.0, 1.0],
        "p_m" => [0.05, 1.0, 0.05]        
    )

    NSGA3_searchspace = OrderedDict(
        "N" => [5, 500, 100],
        "η_cr" => [1.0, 100.0, 1.0],
        "p_cr" =>[0.1, 1, 0.05],
        "η_m" => [0, 50, 100], 
        "partitions" => [0, 500, 10],
    )

    SMS_EMOA_searchspace = OrderedDict(
        "N" => [10, 500, 10],
        "η_cr" => [1.0, 100.0, 1.0],
        "p_cr" => [0.1, 1, 0.05],
        "η_m" =>[1.0, 100.0, 1.0],
        "p_m" => [0.05, 1.0, 0.05],   
        "n_samples" => [1000, 1000000, 500],  
    )

    SPEA2_searchspace = OrderedDict(
        "N" => [10, 500, 10],
        "η_cr" => [1.0, 100.0, 1.0],
        "p_cr" => [0.1, 1.0, 0.05],
        "η_m" => [1.0, 100.0, 1.0],
        "p_m" => [0.05, 1.0, 0.05]   
        )

    MOEAD_DE_searchspace = OrderedDict( 
        "npartitions" => [10, 30, 10], # grows exponentially, must the changed locally
        "F" => [0.1, 1.0, 0.05],
        "CR" => [0.1, 1.0, 0.05],
        "η" => [5.0, 50.0, 1.0],
        "p_m" => [0.05, 1.0, 0.1],
        "δ" =>[0.1, 1.0, 0.05], 
        "s1" => [0.001, 0.1, 0.0005],
        "s2" => [1.0, 50.0, 1.0]
    )


    optuna_sampler_dict = OrderedDict(
        "NSGAIISampler" => optuna.samplers.NSGAIISampler,
        #"CmaEsSampler" => optuna.samplers.CmaEsSampler,
        #"TPESampler" => optuna.samplers.TPESampler,
        #"RandomSampler" => optuna.samplers.RandomSampler,
        #"QMCSampler" => optuna.samplers.QMCSampler,
        #"NSGAIIISampler" => optuna.samplers.NSGAIIISampler,
        #"GPSampler" => optuna.samplers.GPSampler,
        #"BruteForceSampler" => optuna.samplers.BruteForceSampler,
        #"GridSampler" => () -> optuna.samplers.GridSampler(Algorithm_structure.Parameters_ranges)
        )

    sampler_vector =  collect(keys(optuna_sampler_dict))


