using Dates: now
using Crayons

TESTS = true

# Cooling schedule functions
linear_decay(i::Integer, epochs::Integer, yi = 10.0f0^-3.0f0, yf = 10.0f0^-4.0f0) = max(((yf - yi) / epochs) * i + yi, yf)
exp_decay(i::Integer, epochs::Integer, yi = 10.0f0^-3.0f0, yf = 10.0f0^-4.0f0) = max((yi - yf) * exp((yf - yi) * i / epochs) + yf, yf)

# Stopper functions
epochs_max(epoch::Integer, epochs::Integer) = epoch >= epochs
#epochs_stagnant(epoch_since_best::Number, duration::Integer) = length(best_cost) > duration ? best_cost[end] > best_cost[end-duration] : false

cooling_schedule_functions = Dict(
	:linear => linear_decay,
	:exp => exp_decay,
)

search_stopper_functions = Dict(
	:epochs_max => epochs_max,
)

struct SearchStopFunction <: Function
	f::Function
	epochs_parameter::Integer

	function SearchStopFunction(function_symbol::Symbol, epochs_parameter::Integer)
		stop_f = search_stopper_functions[function_symbol]
		return new(stop_f, epochs_parameter)
	end
end
(s::SearchStopFunction)(epoch::Integer) = s.f(epoch, s.epochs_parameter)

CoolingArgs = @NamedTuple begin
	max_epochs::Integer
	init_temperature::Number
	final_temperature::Number
end
struct CoolingFunction <: Function
	f::Function
	args::CoolingArgs

	function CoolingFunction(function_symbol::Symbol, max_epochs::Integer, init_temperature::Number, final_temperature::Number)
		cooling_f = cooling_schedule_functions[function_symbol]
		return new(cooling_f, CoolingArgs((max_epochs, init_temperature, final_temperature)))
	end
end
(c::CoolingFunction)(epoch::Integer) = c.f(epoch, c.args.max_epochs, c.args.init_temperature, c.args.final_temperature)


SearchArgs = @NamedTuple begin
	objective::ObjectiveFunction
	shift::ShiftFunction
	stop::SearchStopFunction
	cooling::CoolingFunction
end

function annealing(architecture::Architecture, search_args::SearchArgs, save_models::Bool = false)
	P(change, temperature) = exp(-change / temperature)

	stop = search_args.stop
	temperature_schedule = search_args.cooling
    shift = search_args.shift
    objective = search_args.objective

	cost_cache = Dict{Vector{Int64}, Float32}()

	init_time = now()
	current_cost, models = objective(architecture)

	best_cost = current_cost
	epoch = 0
	while !stop(epoch)
		epoch = epoch + 1
		temperature = temperature_schedule(epoch)

		neighbor = deepcopy(architecture)
		shift_indices!(shift, neighbor)

		println(Crayon(foreground = :cyan), "Search epoch $(epoch) | Temperature = $(temperature) | Elapsed Time $(now()-init_time)", Crayon(foreground = :default))
		if haskey(cost_cache, neighbor.parameters.index)
			println("            Architecture $(index(neighbor)) Already computed")
			cost_neighbor = cost_cache[neighbor.parameters.index]
		else
			println("Training an Architecture $(index(neighbor)) model")
			cost_neighbor, trial_models = objective(neighbor)
			if save_models
				append!(models, trial_models)
			end
			cost_cache[index(neighbor)] = cost_neighbor
		end

		delta = cost_neighbor - current_cost

		if (delta < 0) | (P(delta, temperature) > rand())
			architecture_new = neighbor
			current_cost = cost_neighbor
			if current_cost < best_cost
				best_cost = current_cost
			end
			println("     Accepted new solution $(index(architecture_new)) with cost: $(current_cost) | Best cost: $(best_cost)")
		else
			architecture_new = architecture
			println("Kept the existing solution $(index(architecture_new)) with cost: $(current_cost) | Best cost: $(best_cost) | Neighbor cost: $(cost_neighbor)")
		end
	end

	return best_cost, models
end

function TEST_annealing()
	in_shapes = [(6,)]
	out_shapes = [(6,)]
	num_samples = 8
	data = (rand(Float32, in_shapes[1] .+ (0, num_samples - 1)), rand(Float32, out_shapes[1] .+ (0, num_samples - 1)),
		rand(Float32, in_shapes[1] .+ (0, num_samples - 1)), rand(Float32, out_shapes[1] .+ (0, num_samples - 1)))

	loss = Flux.mae

	λ = RegularizationFunction(:l1, 0.1f0)
	η = LearnRateFunction(:exp, 100, 0.01f0, 0.002f0)
	train_stopper = TrainStopFunction(:epochs_max, 1)

	train_sequence = [TrainingArgs((num_samples, 0.1f0, loss, η, λ, train_stopper)) for _ in 1:1]
	num_trials = 1

	format = :fixed
	space = [[1], [4, 5, 6, 7, 8], [Flux.relu, Flux.leakyrelu]]
	symbols = [:length, :width, :activation]

	architecture = Architecture(space, symbols, format, in_shapes, out_shapes)
	objective = backprop_objective_generator(data, train_sequence, num_trials)
	shift = ShiftFunction(length(space), Exponential(1.0), 1.0)

	stopper_args = SearchStopFunction(:epochs_max, 5)
	cooling_args = CoolingFunction(:exp, 10, 0.5, 0.01)

	search_args = SearchArgs((objective, shift, stopper_args, cooling_args))

	best_cost, models = annealing(architecture, search_args, false)

	return true
end

if TESTS
	@assert TEST_annealing()
end
