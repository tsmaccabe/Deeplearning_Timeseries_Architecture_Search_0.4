## Activate environment
using Pkg
const PWD = replace(pwd(), "\\" => "/")
Pkg.activate(PWD)

## Locate and use local modules
const MODULES_PATH = PWD*"/modules/"
if !(MODULES_PATH in LOAD_PATH)
    push!(LOAD_PATH, MODULES_PATH)
end

include(MODULES_PATH*"TrainingUtils.jl")
include(MODULES_PATH*"ModelUtils.jl")
include(MODULES_PATH*"DataUtils.jl")
include(MODULES_PATH*"FileManagement.jl")
include(MODULES_PATH*"Parametrizers.jl")
include(MODULES_PATH*"ShiftFunctions.jl")
include(MODULES_PATH*"Architectures.jl")
include(MODULES_PATH*"ObjectiveFunctions.jl")
include(MODULES_PATH*"Search.jl")

## Load config
include("load_config.jl")

## Finished initializing
println("Environment initialized: "*PWD)