include(MODULES_PATH*"CreateLorenzData.jl")

function main()
    num_timesteps = 50000
    save_lorenz(DATA_RAW_DIR, num_timesteps, num_timesteps)
    data_raw = load_lorenz(1, num_timesteps)[1]
    u = data_raw["u"]

    num_nodes = 30
    in_shapes = [(num_nodes,)]; out_shapes = [(num_nodes,)]
    num_samples = 1000

    data_dict = pair_data(u, 10, num_samples, num_nodes, 1)
    x = data_dict["x"]; y = data_dict["y"]

    train_pct = 0.8f0
    data = split_data(x, y, train_pct)
    train_samples = size(data[1], length(axes(data[1])))
    test_samples = num_samples - train_samples

    max_epochs_train = 5
    learn_rate_params = (100, 0.05f0, 0.005f0)
    regularization_param = 0.01f0

    loss = Flux.mae
    reg = (model) -> regularization_functions[:l1](model, regularization_param)
    learn_rate = (epoch) -> learn_rate_functions[:linear](epoch, learn_rate_params...)
    train_stopper = (losses_test, losses_train) -> train_stopper_functions[:epochs_max](losses_test, losses_train, max_epochs_train)
        
    train_sequence = [TrainingArgs(test_samples, 0.1f0, loss, learn_rate, reg, train_stopper) for _ in 1:1]
    num_trials = 2

    format = :fixed
    space = [[1, 2, 3], [5, 6, 7], [Flux.relu, Flux.leakyrelu]]
    symbols = [:length, :width, :activation]

    architecture = Architecture(space, symbols, format, in_shapes, out_shapes)
    objective = backprop_objective_generator(data, train_sequence, num_trials)
    shift = ShiftFunction(length(space), Exponential(1.), 1.)

    stopper_args = StopperArgs(:epochs_max, 7)
    cooling_args = CoolingArgs(:exp, 10, 0.5, 0.01)

    search_args = SearchArgs(objective, shift, stopper_args, cooling_args)
        
    best_cost, models = annealing(architecture, search_args, false)

    return best_cost, models
end

best_cost, models = main()