TESTS = true

struct ObjectiveFunction <: Function
	f::Function
	stored_vars::Tuple
	function ObjectiveFunction(f::T, stored_vars::Tuple) where T <: Function
		return new(f, stored_vars)
	end
	function ObjectiveFunction(f::T, stored_vars...) where T <: Function
		return new(f, stored_vars)
	end
end
(objective::ObjectiveFunction)(architecture::Architecture) = objective.f(architecture, objective.stored_vars...)

function backprop_trials(architecture::Architecture, data::NTuple{4, Array{Float32}}, train_args::Vector{TrainingArgs}, num_trials::Integer, keep_models::Bool=false)
	losses_test = zeros(Float32, num_trials)
	models = keep_models ? Vector{Chain}([]) : nothing
	for trials_losses_index in 1:num_trials
		println("Architecture $(values(architecture)) | Trial $(trials_losses_index)")
		loss_test, _, model = train_sequence(architecture(), data, train_args, keep_models)
		losses_test[trials_losses_index] = loss_test[end]
		if keep_models push!(models, model) end
	end
	
	return mean(losses_test), models
end

backprop_objective_function(data, train_args, num_trials) = ObjectiveFunction(backprop_trials, (data, train_args, num_trials))

function TEST_ObjectiveFunction()
	in_shapes = [(6,)]; out_shapes = [(6,)]
	num_samples = 8
	data = (rand(Float32, in_shapes[1].+(0, num_samples-1)), rand(Float32, out_shapes[1].+(0, num_samples-1)),
			rand(Float32, in_shapes[1].+(0, num_samples-1)), rand(Float32, out_shapes[1].+(0, num_samples-1)))

	max_epochs_train = 4
	learn_rate_params = (100, 0.01f0, 0.002f0)
	regularization_param = 0.1f0

	loss = Flux.mae
	reg = (model) -> regularization_functions[:l1](model, regularization_param)
	learn_rate = (epoch) -> learn_rate_functions[:linear](epoch, learn_rate_params...)
	stopper = (losses_test, losses_train) -> train_stopper_functions[:epochs_max](losses_test, losses_train, max_epochs_train)
	
	train_sequence = [TrainingArgs(2, 0.1f0, loss, learn_rate, reg, stopper) for _ in 1:1]
	num_trials = 2

	format = :fixed
    space = [[1, 2, 3], [1, 2, 3], [Flux.relu, Flux.leakyrelu]]
    symbols = [:length, :width, :activation]
    architecture = Architecture(space, symbols, format, in_shapes, out_shapes)

	test_ObjectiveFunction = backprop_objective_function(data, train_sequence, num_trials)
	println(architecture.key)
    #@assert architecture.key == string(Profile(model_formats[format], in_shapes, out_shapes))
    
	trial_loss, models = test_ObjectiveFunction(architecture)
	println(trial_loss)

	return true
end

if TESTS
	@assert TEST_ObjectiveFunction()
end
