import Base: setindex!

abstract type AbstractArchitecture end

TESTS = true

struct Profile
    format::Function
    in_shapes::ShapeVector
    out_shapes::ShapeVector
end
(profile::Profile)(param_values) = profile.format(profile.in_shapes, profile.out_shapes, param_values)

function make_key(profile::Profile)
    in_shapes_str = join([join(shape, ", ") for shape in profile.in_shapes], "; ")
    out_shapes_str = join([join(shape, ", ") for shape in profile.out_shapes], "; ")
    
    return Symbol("format: $(profile.format), in_shapes: [$(in_shapes_str)], out_shapes: [$(out_shapes_str)]")
end

mutable struct Architecture <: AbstractArchitecture
    key::Symbol
    parameters::Parametrizer
    profile::Profile
    function Architecture(param_space, param_symbols, format_key, in_shapes, out_shapes)
        format = model_formats[format_key]
        profile = Profile(format, in_shapes, out_shapes)
        key = make_key(profile)
        parameters = Parametrizer(param_space, param_symbols)
        return new(key, parameters, profile)
    end
end
in_shapes(architecture::Architecture) = architecture.profile.in_shapes
out_shapes(architecture::Architecture) = architecture.profile.out_shapes
values(architecture::Architecture) = values(architecture.parameters)
index(architecture::Architecture) = architecture.parameters.index
(architecture::Architecture)() = architecture.profile(values(architecture))

function setindex!(architecture::Architecture, new_index::Vector{Int})
    setindex!(architecture.parameters, new_index)
end

function shift_indices!(shift_function::ShiftFunction, architecture::Architecture)
    shift_indices!(shift_function, [architecture.parameters])
end
function shift_indices!(shift_function::ShiftFunction, architectures::Vector{Architecture})
    shift_indices!(shift_function, [architecture.parameters for architecture in architectures])
end

mutable struct CompositeArchitecture <: AbstractArchitecture
    key::String
    architectures::Dict{String, AbstractArchitecture} # Maps each component architecture's key to the architecture itself
    execution_layers::Vector{Vector{Symbol}}
    parameters::Parametrizer
    profile::Profile
    function CompositeArchitecture(architectures_list::Vector{AbstractArchitecture}, execution_order::Vector{Vector{Integer}})
        architectures = Dict([architecture.key => architecture for architecture in architectures_list])

        execution_layers = [[architecture.key for architecture in architectures[layer_architecture_indices]] for layer_architecture_indices in execution_order]


        key = make_key(profile)
        parameters = Parametrizer([architecture.parametrizer for architecture in architectures_list])
        return new(key, architectures, execution_layers, parameters, profile)
    end
end

function TEST_Architecture()
    format = :fixed; in_shapes = [(50, 1), (50, 1)]; out_shapes = [(50, 1), (50, 1)]
    space = [[1, 2, 3], [1, 2, 3], [Flux.mae, Flux.mse]]
    symbols = [:length, :width, :activation]
    test_Architecture = Architecture(space, symbols, format, in_shapes, out_shapes)
    @assert test_Architecture.key == make_key(Profile(model_formats[format], in_shapes, out_shapes))
    return test_Architecture
end

if TESTS
    test_Architecture = TEST_Architecture()
    println(test_Architecture.key)
end