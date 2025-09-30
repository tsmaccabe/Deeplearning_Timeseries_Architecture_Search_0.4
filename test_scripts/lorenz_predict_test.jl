include(MODULES_PATH*"CreateLorenzData.jl")

function main()
    num_timesteps = 400000
    save_lorenz(DATA_RAW_DIR, num_timesteps, num_timesteps)
    data_raw = load_lorenz(1, num_timesteps)[1]
    u = data_raw["u"]

    sample_timesteps = 16
    num_nodes = 3*sample_timesteps
    in_shapes = [(num_nodes,)]; out_shapes = [(num_nodes,)]
    num_samples = 2^14

    data_dict = pair_data(u, 10, num_samples, num_nodes, sample_timesteps)
    x = data_dict["x"]; y = data_dict["y"]

    train_pct = 0.8f0
    data = split_data(x, y, train_pct)
    train_samples = size(data[1], length(axes(data[1])))
    test_samples = num_samples - train_samples

    loss = Flux.mae
	λ = RegularizationFunction(:l2, 0.1f0)
	η = LearnRateFunction(:linear, 50, 0.01f0, 0.002f0)
	train_stop = TrainStopFunction(:epochs_stagnant, 25)

    batch_size = 128
    train_sequence = [TrainingArgs(batch_size, 0.1f0, loss, η, λ, train_stop) for _ in 1:1]
    num_trials = 1

    format = :fixed
    space = [collect(1:30), collect(1:30), [Flux.relu, Flux.leakyrelu]]
    symbols = [:length, :width, :activation]

    architecture = Architecture(space, symbols, format, in_shapes, out_shapes)
    objective = backprop_objective_function(data, train_sequence, num_trials)
    shift = ShiftFunction(1)

	search_stopper = SearchStopFunction(:epochs_max, 100)
	cooling = CoolingFunction(:linear, 0.5f0, 0.01f0, 70)

    search_args = SearchArgs((objective, shift, search_stopper, cooling) )
        
    best_cost, models = annealing(architecture, search_args, false)

    return best_cost, models
end

best_cost, models = main()