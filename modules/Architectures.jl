import Base: setindex!

TESTS = true

struct Profile
    format::Function
    in_shapes::ShapeVector
    out_shapes::ShapeVector
end
(profile::Profile)(param_values) = profile.format(profile.in_shapes, profile.out_shapes, param_values)

mutable struct Architecture
    key::String
    parameters::Parametrizer
    profile::Profile
    function Architecture(param_space, param_symbols, format_key, in_shapes, out_shapes)
        format = model_formats[format_key]
        profile = Profile(format, in_shapes, out_shapes)
        key = string(profile)
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

function TEST_Architecture()
    format = :fixed; in_shapes = [(50, 1)]; out_shapes = [(50, 1)]
    space = [[1, 2, 3], [1, 2, 3], [Flux.mae, Flux.mse]]
    symbols = [:length, :width, :activation]
    test_Architecture = Architecture(space, symbols, format, in_shapes, out_shapes)
    @assert test_Architecture.key == string(Profile(model_formats[format], in_shapes, out_shapes))
    return true
end

if TESTS
    @assert TEST_Architecture()
end