include(MODULES_PATH*"CreateLorenzData.jl")

function main()
    num_timesteps = 50000
    save_lorenz(DATA_RAW_DIR, num_timesteps, num_timesteps)
    data_raw = load_lorenz(1, num_timesteps)[1]
    u = data_raw["u"]

    num_nodes = 30
    in_shapes = [(num_nodes,)]; out_shapes = [(num_nodes,)]
    num_samples = 100

    data_dict = pair_data(u, 10, num_samples, num_nodes, 1)
    x = data_dict["x"]; y = data_dict["y"]

    train_pct = 0.8f0
    data = split_data(x, y, train_pct)
    train_samples = size(data[1], length(axes(data[1])))
    test_samples = num_samples - train_samples

    loss = Flux.mae
	λ = RegularizationFunction(:l1, 0.1f0)
	η = LearnRateFunction(:linear, 5, 0.01f0, 0.002f0)
	train_stop = TrainStopFunction(:epochs_max, 3)

    train_sequence = [TrainingArgs((test_samples, 0.1f0, loss, η, λ, train_stop)) for _ in 1:1]
    num_trials = 1

    format = :fixed
    space = [[1, 2, 3], [5, 6, 7], [Flux.relu, Flux.leakyrelu]]
    symbols = [:length, :width, :activation]

    architecture = Architecture(space, symbols, format, in_shapes, out_shapes)
    objective = backprop_objective_generator(data, train_sequence, num_trials)
    shift = ShiftFunction(length(space), Exponential(1.), 1.)

	search_stopper = SearchStopFunction(:epochs_max, 5)
	cooling = CoolingFunction(:exp, 10, 0.5, 0.01)

    search_args = SearchArgs((objective, shift, search_stopper, cooling) )
        
    best_cost, models = annealing(architecture, search_args, false)

    return best_cost, models
end

best_cost, models = main()