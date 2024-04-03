using Flux: Chain
using JLD2, FileIO, CSV, Dates

# Each Dict entry has a key that corresponds to a list of file paths that collectively 
# contain the key's corresponding data set
DATA_REGISTRY = Dict()
MODEL_REGISTRY = Dict()

const STORAGE_DIR = PWD*"/storage/"
const DATA_DIR = STORAGE_DIR*"data sets/"
const DATA_RAW_DIR = DATA_DIR*"raw/"
const MODELS_DIR = STORAGE_DIR*"models/"
const MODELS_TRAINED_DIR = MODELS_DIR*"trained/"
const RESULTS_DIR = STORAGE_DIR*"results/"

function log_call(function_name::String, key::String, path::String)
    # Ensure the log file exists, creating it with headers if it doesn't
    if !isfile(STORAGE_DIR*"log.csv")
        CSV.write(STORAGE_DIR*"log.csv", [(function_name = "Function Name", key = "Key", path = "Path", time = "Time")], append = false)
    end

    # Append the new log entry
    CSV.write(STORAGE_DIR*"log.csv", [(function_name = function_name, key = key, path = path, time = Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))], append = true)
end

# Instead, call register_data multiple times
#=function register_data(key::String, paths::Vector{String})
    DATA_REGISTRY[key] = paths
    for path in paths
        log_call("register_data", key, path)
    end
end=#
function register_data(key::String, path::String)
    DATA_REGISTRY[key] = path
    log_call("register_data", key, path)
end

function register_model(key::String, path::String)
    MODEL_REGISTRY[key] = path
    log_call("register_model", key, path)
end

function load_data(key::String)
    path = DATA_REGISTRY[key]
    log_call("load_data", key, path)
    return load_object(path*key*".jld2")
end
function load_datasets(keys::Vector{String})
    return [ load_data(key) for key in keys ]
end

function save_data(data::Dict, key::String)
    path = DATA_REGISTRY[key]
    save_object(path*key*".jld2", data)
    log_call("save_data", key, path)
end
function save_datasets(datasets::Vector{Dict}, keys::Vector{String})
    for i in axes(keys)[1] save_data(datasets[i], keys[i]) end
end

function load_model(key::String)
    path = MODEL_REGISTRY[key]
    log_call("load_model", key, path)
    return load_object(path*key*".jld2")
end
function load_models(keys::Vector{String})
    return [ load_model(key) for key in keys ]
end

function save_model(model::Chain, key::String)
    path = MODEL_REGISTRY[key]
    save_object(path*key*".jld2", model)
    log_call("save_model", key, path)
end
function save_models(models::Vector{Chain}, keys::Vector{String})
    for i in axes(keys)[1] save_model(models[i], keys[i]) end
end