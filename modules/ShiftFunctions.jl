using Distributions

TESTS = true

function uniform_directions(n::Int, count::Int)
	# Generate a matrix of random numbers
	random_matrix = randn(n, count)

	# Normalize each column to get unit vectors
	norms = sqrt.(sum(random_matrix .^ 2, dims = 1))
	directions = random_matrix ./ norms

	return directions
end

struct ShiftFunction <: Function
	decay::Distribution
	direction::Function
	radial_scale::Number
	function ShiftFunction(numeric_dimensions::Int, radial_decay::Distribution, radial_scale::Number)
		direction_fcn = (num_directions) -> uniform_directions(numeric_dimensions, num_directions)
		return new(radial_decay, direction_fcn, radial_scale)
	end
end
(shift_function::ShiftFunction)() :: Vector{T} where T = round.(Int, shift_function.radial_scale * rand(shift_function.decay) * shift_function.direction(1))[:, 1]
#(shift_function::ShiftFunction)(parametrizers::Vector{Parametrizer}) = shift_function(length(parametrizers[1].index))
function shift_indices!(shift_function::ShiftFunction, parametrizers::Vector{Parametrizer})
    shift = zeros(length(parametrizers))
    for parametrizer in parametrizers
        shift = shift_function()
        while all([elem == 0 for elem in shift])
            shift = shift_function()
        end
        setindex!(parametrizer, parametrizer.index + shift_function())
    end
    return shift
end

function TEST_ShiftFunction(dimensions::Integer=15, num_parametrizers::Integer=5)
    syms = [Symbol(string("a", i)) for i in 1:dimensions]
    space = Vector{Vector}([collect(i^2:i^2+9) for i in 1:dimensions])
    parametrizers = [Parametrizer(space, syms) for _ in 1:num_parametrizers]

    test_ShiftFunction = ShiftFunction(dimensions, Exponential(1.), 1)
    shift_indices!(test_ShiftFunction, parametrizers)

    for _ in 1:10
        shift_indices!(test_ShiftFunction, parametrizers)
    end

    return true
end

if TESTS
    @assert TEST_ShiftFunction()
end