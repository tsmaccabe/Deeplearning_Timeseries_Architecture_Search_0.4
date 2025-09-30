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

function uniform_direction(n::Int)
	# Generate a matrix of random numbers
	random_matrix = randn(n)

	# Normalize each column to get unit vectors
	norms = sqrt.(sum(random_matrix .^ 2, dims = 1))
	directions = random_matrix ./ norms

	return directions
end

struct ShiftFunction <: Function
	range::Integer
end
(shift_function::ShiftFunction)(num_dimensions::Integer) :: Vector{T} where T = [rand(-shift_function.range:shift_function.range) for _ in 1:num_dimensions]
function shift_indices!(shift_function::ShiftFunction, parametrizers::Vector{Parametrizer})
    shift = zeros(length(parametrizers))
    for parametrizer in parametrizers
        shift = shift_function(length(parametrizer))
        println("Shift: ", shift, ", indices before: ", parametrizer.index, ", indices after: ", mod.(parametrizer.index + shift, length(parametrizer)), ", length: ", length(parametrizer))
        setindex!(parametrizer, mod.(parametrizer.index + shift, length(parametrizer)))
    end
    return shift
end

function TEST_ShiftFunction(dimensions::Integer=15, num_parametrizers::Integer=5)
    syms = [Symbol(string("a", i)) for i in 1:dimensions]
    space = Vector{Vector}([collect(i^2:i^2+9) for i in 1:dimensions])
    parametrizers = [Parametrizer(space, syms) for _ in 1:num_parametrizers]

    test_ShiftFunction = ShiftFunction(0)
    shift_indices!(test_ShiftFunction, parametrizers)

    for _ in 1:10
        shift_indices!(test_ShiftFunction, parametrizers)
    end

    return true
end

if TESTS
    @assert TEST_ShiftFunction()
end