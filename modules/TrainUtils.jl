using CUDA, Optimisers, Flux, Printf
using Dates: now

active_device = CUDA.functional() ? gpu : cpu

# Learning rate functions
linear_decay(i::Integer, epochs::Integer, yi = 10.0f0^-3.0f0, yf = 10.0f0^-4.0f0) = max(((yf - yi) / epochs) * i + yi, yf)
exp_decay(i::Integer, epochs::Integer, yi = 10.0f0^-3.0f0, yf = 10.0f0^-4.0f0) = max((yi - yf) * exp((yf - yi) * i / epochs) + yf, yf)

# Regularization functions
l1_regularization(layers, λ = 0.001f0) = λ * sum(sum(abs, p) for l in layers for p in Flux.params(l))
l2_regularization(layers, λ = 0.0001f0) = λ * sum(sum(p .^ 2) for l in layers for p in Flux.params(l))

# Stopper functions
epochs_max(loss_t, loss_e, epochs::Integer) = length(loss_e) >= epochs
epochs_stagnant(loss_t, loss_e, duration::Integer) = length(loss_e) > duration ? loss_e[end] > loss_e[end-duration] : false

loss_functions = Dict(
	:mae => Flux.Losses.mae,
	:mse => Flux.Losses.mse,
	:cross_entropy => Flux.Losses.crossentropy,
)
learn_rate_functions = Dict(
	:linear => linear_decay,
	:exp => exp_decay,
)
regularization_functions = Dict(
	:l1 => l1_regularization,
	:l2 => l2_regularization,
)
train_stopper_functions = Dict(
	:epochs_max => epochs_max,
	:epochs_stagnant => epochs_stagnant,
)

function reset_parameters!(model)
	for layer in model
		if (layer isa Dense) || (layer isa Conv)
			# Reset weights with Glorot uniform distribution
			layer.weight .= Flux.glorot_uniform(size(layer.weight)...)
			# Reset biases to zero
			layer.bias .= zeros(size(layer.bias))
		elseif layer isa Chain
			# Recursively reset parameters for nested chains
			reset_parameters!(layer)
			# Add more conditions here for other layer types if necessary
		end
	end
end

function set_dropout_rate!(model, drop_rate::Float32)
	if model isa Flux.Dropout
		model.p = drop_rate
	end

	if model isa Flux.Chain
		for layer in model.layers
			set_dropout_rate!(layer, drop_rate)
		end
	end
end

LearnRateArgs = @NamedTuple begin
	final_epoch::Integer
	init_value::Number
	final_value::Number
end
struct LearnRateFunction <: Function
	f::Function
	args::LearnRateArgs
	function LearnRateFunction(function_symbol::Symbol, final_epoch::Integer, init_value::Number, final_value::Number)
		return new(learn_rate_functions[function_symbol], LearnRateArgs((final_epoch, init_value, final_value)))
	end
end
(η::LearnRateFunction)(epoch::Integer) = η.f(epoch, η.args...)

struct RegularizationFunction <: Function
	f::Function
	magnitude::Number
	function RegularizationFunction(function_symbol::Symbol, magnitude::Number)
		return new(regularization_functions[function_symbol], magnitude::Number)
	end
end
(λ::RegularizationFunction)(model) = λ.f(model, λ.magnitude)

struct TrainStopFunction <: Function
	f::Function
	epochs_parameter::Integer
	function TrainStopFunction(function_symbol::Symbol, epochs_parameter::Integer)
		return new(train_stopper_functions[function_symbol], epochs_parameter)
	end
end
(stop::TrainStopFunction)(loss_t, loss_e) = stop.f(loss_t, loss_e, stop.epochs_parameter)

TrainingArgs = @NamedTuple begin
	batch_size::Integer
	drop_rate::Float32
	loss::Function
	η::LearnRateFunction
	λ::RegularizationFunction
	stop::TrainStopFunction
end

function train_loop!(model::Chain, dataset::NTuple{4, Array{Float32}}, p_train::TrainingArgs)
	(batch_size, drop_rate, loss, η, λ, stop) = (p_train.batch_size, p_train.drop_rate, p_train.loss, p_train.η, p_train.λ, p_train.stop)
	(data, targets, test_data, test_targets) = (dataset...,)

	n_train_samples = axes(data)[end][end]
	n_test_samples = axes(test_data)[end][end]
	r_train_test = n_train_samples / n_test_samples

	init_time = now()

	opt = Flux.setup(Optimisers.Adam(), model)
	set_dropout_rate!(model, drop_rate)

	loss_regularized(m, x, y) = loss(m(x), y) + λ(m)

	L_test = Float32[]
	L_train = Float32[]
	epoch = 0
	best_test_loss = Inf32
	best_params = nothing
	while !stop(L_train, L_test)
		epoch += 1
		Optimisers.adjust!(opt, η(epoch))

		# Perform backpropagation
		train_loader = Flux.DataLoader((data, targets), batchsize = batch_size, shuffle = true)
		current_train_loss_gpu = 0.0f0 |> active_device
		for (x_batch, y_batch) in train_loader
			x_batch = x_batch |> active_device
			y_batch = y_batch |> active_device

			grad = gradient(loss_regularized, model, x_batch, y_batch)[1]
			Flux.Optimise.update!(opt, model, grad)

			Flux.testmode!(model)
			current_train_loss_gpu += loss_regularized(model, x_batch, y_batch)
			Flux.trainmode!(model)
		end
		current_train_loss = current_train_loss_gpu |> cpu

		# Calculate test loss
		test_loader = Flux.DataLoader((test_data, test_targets), batchsize = batch_size, shuffle = true)
		current_test_loss_gpu = 0.0f0 |> active_device
		for (x_batch, y_batch) in test_loader
			x_batch = x_batch |> active_device
			y_batch = y_batch |> active_device

			Flux.testmode!(model)
			current_test_loss_gpu += loss_regularized(model, x_batch, y_batch)
			Flux.trainmode!(model)
		end
		current_test_loss = (current_test_loss_gpu * r_train_test) |> cpu

		push!(L_train, current_train_loss)
		push!(L_test, current_test_loss)

		if current_test_loss < best_test_loss
			best_test_loss = current_test_loss
			best_params = Flux.params(model)
		end
	end

	formatted_output = @sprintf("epochs = %d | Train Loss = %.6f, Test Loss = %.6f", epoch, L_train[end], L_test[end])
	println(formatted_output, " | Training Time $(init_time - now())")

	if best_params !== nothing
		Flux.loadparams!(model, best_params)
	end

	return L_test, L_train
end

function train_sequence_cpu!(model::Chain, data, subsequences::Vector{TrainingArgs}, keep_models::Bool = false)
	# CPU model
	L_test = []
	L_train = []
	for p_train in subsequences
		L_test_this, L_train_this = train_loop!(model, data, p_train)

		L_test = vcat(L_test, L_test_this)
		L_train = vcat(L_train, L_train_this)
	end
	model = keep_models ? model : nothing
	return L_test, L_train, model
end
function train_sequence_gpu(model_cpu::Chain, data, subsequences::Vector{TrainingArgs}, keep_models::Bool = false)
	# GPU model
	L_test = []
	L_train = []
	model = model_cpu |> gpu
	for p_train in subsequences
		L_test_this, L_train_this = train_loop!(model, data, p_train)

		L_test = vcat(L_test, L_test_this)
		L_train = vcat(L_train, L_train_this)
	end
	model_cpu = keep_models ? model |> cpu : nothing
	return L_test, L_train, model_cpu
end
function train_sequence(model, data, subsequences, keep_models::Bool = false)
	return CUDA.functional() ? train_sequence_gpu(model, data, subsequences, keep_models) : train_sequence_cpu!(model, data, subsequences, keep_models)
end
