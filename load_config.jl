# Function to parse a tuple from a string
function parse_tuple(tuple_str)
    eval(Meta.parse(tuple_str))
end

# Read the configuration file
using IniFile
config_file = "config.ini"
ini = read(Inifile(), config_file)

# Architecture Settings
architecture_type_str = get(ini, "Architecture Settings", "architecture_type")
architecture_type = parse_tuple(architecture_type_str)
activation_str = get(ini, "Architecture Settings", "activation")
activation = parse_tuple(activation_str)

# Search Settings
stop_condition_search_str = get(ini, "Search Settings", "stop_condition_search")
stop_condition_search = parse_tuple(stop_condition_search_str)
cool_schedule_str = get(ini, "Search Settings", "cool_schedule")
cool_schedule = parse_tuple(cool_schedule_str)
shift_settings_str = get(ini, "Search Settings", "shift_settings")
shift_settings = parse_tuple(shift_settings_str)
trials_per_state = parse(Int, get(ini, "Search Settings", "trials_per_state"))

# Training Settings
stop_condition_train_str = get(ini, "Training Settings", "stop_condition_train")
stop_condition_train = parse_tuple(stop_condition_train_str)

learn_rate_str = get(ini, "Training Settings", "learn_rate")
learn_rate = parse_tuple(learn_rate_str)
loss_function_str = get(ini, "Training Settings", "loss_function")
loss_function_symbol = Symbol(loss_function_str)
dropout_rate = parse(Float32, get(ini, "Training Settings", "dropout_rate"))
regularization_str = get(ini, "Training Settings", "regularization")
regularization = parse_tuple(regularization_str)

# Use Configuration Data to Define Objects
architecture_search_space = [architecture_type[2:end]..., [activation...]]
architecture_vars = (architecture_search_space, [:length, :width, :activation], architecture_type[1])

λ = RegularizationFunction(regularization...)
println(learn_rate)
η = LearnRateFunction(learn_rate...)
loss = if loss_function_symbol == :mae
    Flux.mae
elseif loss_function_symbol == :mse
    Flux.mse
elseif loss_function_symbol == :cross_entropy
    Flux.cross_entropy
end
train_stopper = TrainStopFunction(stop_condition_train...)
println(stop_condition_train)
search_stopper = SearchStopFunction(stop_condition_search...)
cooling = CoolingFunction(cool_schedule...)
numeric_dimensions = prod(prod(architecture_type[i]) for i in axes(architecture_type)[1][2:end])
state_shift = ShiftFunction(numeric_dimensions, shift_settings[2])

dropout_rate = regularization[2]


println("Configuration")
println("Architecture Type: ", architecture_type)
println("Activation: ", activation)
println("Stop Condition (search): ", stop_condition_search)
println("Stop Condition (train): ", stop_condition_train)
println("Shift Settings: ", shift_settings)
println("Trials Per State: ", trials_per_state)
println("Regularization: ", regularization)
println("Learning Rate: ", learn_rate)
println("Dropout Rate: ", dropout_rate)
