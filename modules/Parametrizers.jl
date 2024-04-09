#module Parametrizers

TESTS = true

import Base: length, setindex!

export Parametrizer, SingleParametrizer, AbstractParametrizer, FunctionParameter
export length, values, setindex!

struct FunctionParameter <: Function
	f::Function
	function FunctionParameter(f::T) where T <: Function
		if T == FunctionParameter
			return new(f.f)
		else
			return new(f)
		end
	end
end
(function_parameter::FunctionParameter)(x...) = function_parameter.f(x...)

#values(space::Vector{Vector{T}}, index::Vector{Int}) where T = [space[space_index][index[space_index]] for space_index in axes(index)[1]]
values(space::Vector{Vector{T}}, index::Vector{Int}) where T = typejoin([space[space_index][index[space_index]] for space_index in axes(index)[1]])
values(space::Vector{Vector}, index::Vector{Int}) = typejoin([space[space_index][index[space_index]] for space_index in axes(index)[1]])
function parent_type(values::Vector)
	type = typejoin(typeof.(values)...)
	if any([(typeof(value) != FunctionParameter) & (typeof(value) <: Function) for value in values])
		type = FunctionParameter
	end
	return type
end

abstract type AbstractParametrizer end
"""
Common Fields:
	index::Vector
	values::Vector
Common Methods:
	Base.length
	setindex!
	space::Vector{Vector}
	values::Vector"""

mutable struct SingleParametrizer{T} <: AbstractParametrizer
	index::Vector{Int}
	values::Vector{T}
	symbols::Vector{Symbol}
	space::Vector{Vector{T}}
	type::DataType
	function SingleParametrizer(space::Vector{Vector{T}}, symbols::Vector{Symbol}) where T
		if length(space) != length(symbols)
			throw(ArgumentError("the space vector and symbols vector must be the same length."))
		end
		index = ones(Int, length(space))
		values = [space[i][index[i]] for i in axes(index)[1]]
		type = typeof(values[1])
		return new{type}(index, values, symbols, space, type)
	end
	function SingleParametrizer(space::Vector{Vector}, symbols::Vector{Symbol})
		type = parent_type([value for domain in space for value in domain])
		if (type <: Function) & (type != FunctionParameter)
			space = [FunctionParameter.(domain) for domain in space]
		end
		return SingleParametrizer(Vector{type}.(space), symbols)
	end
end
length(parametrizer::SingleParametrizer)::Int = length(parametrizer.index)
values(parametrizer::SingleParametrizer{T}) where T = parametrizer.values::Vector{T}
get_space(parametrizer::SingleParametrizer{T}) where T = parametrizer.space::Vector{Vector{T}}

function setindex!(parametrizer::SingleParametrizer{T}, new_index::Vector{Int}) where T
	parametrizer.index = [mod(index_dimension + 1, length(domain)) + 1 for (index_dimension, domain) in zip(new_index, parametrizer.space)]
	parametrizer.values = values(parametrizer.space, parametrizer.index)
end

function TEST_SingleParametrizer(max_test_domains = 10)
	for test_domains in 1:max_test_domains
		test_SingleParametrizer = SingleParametrizer([collect(1:5) for _ in 1:test_domains], Symbol.(1:test_domains))
		@assert length(test_SingleParametrizer.index) == length(test_SingleParametrizer.values) == test_domains
		test_SingleParametrizer_2 = deepcopy(test_SingleParametrizer)
		setindex!(test_SingleParametrizer_2, test_SingleParametrizer.index + ones(Int, test_domains))
		setindex!(test_SingleParametrizer, test_SingleParametrizer.index + ones(Int, test_domains))
		@assert test_SingleParametrizer_2.index == test_SingleParametrizer.index
		@assert values(test_SingleParametrizer) == values(test_SingleParametrizer.space, test_SingleParametrizer.index)
	end
	try
		SingleParametrizer([zeros(Int, 5) for _ in 1:max_test_domains], Symbol.(1:max_test_domains-1))
		@assert false
	catch err
		@assert err isa ArgumentError
		@assert err.msg == "the space vector and symbols vector must be the same length."
	end
	return true
end

if TESTS
	@assert TEST_SingleParametrizer()
end

mutable struct Parametrizer <: AbstractParametrizer
	index::Vector{Int}
	single_types::Vector{SingleParametrizer}
	symbols::Vector{Symbol}
	types::Vector{DataType}
	function Parametrizer(space::Vector{Vector}, symbols::Vector{Symbol})
		if length(space) != length(symbols)
			throw(ArgumentError("the space vector and symbols vector must be the same length."))
		end
		if symbols != unique(symbols)
			throw(ArgumentError("each symbol in symbols must be unique."))
		end
		types = unique([parent_type(domain) for domain in space])
		single_parametrizers = SingleParametrizer[]
		for type in types
			type_idc = findall([parent_type(domain) == type for domain in space])
			type_symbols = symbols[type_idc]
			type_subspace = [type.(domain) for domain in space[type_idc]]
			type_parametrizer = SingleParametrizer(type_subspace, type_symbols)
			push!(single_parametrizers, type_parametrizer)
		end
		index = vcat([dimension for single in single_parametrizers for dimension in single.index])
		return new(index, single_parametrizers, symbols, types)
	end
end
length(parametrizer::Parametrizer) = sum([length(single.index) for single in parametrizer.single_types])
values(parametrizer::Parametrizer)::Vector = vcat([values(single) for single in parametrizer.single_types]...)
function get_space(parametrizer::Parametrizer)::Vector{Vector}
	space = [Vector{single.type}(domain) for single in parametrizer.single_types for domain in single.space]
	common_type = typejoin([parent_type(domain) for domain in space]...)
	return Vector{Vector{<:common_type}}(space)
end

function Parametrizer(parametrizers::Vector{Parametrizer})
	space = [domain for parametrizer in parametrizers for domain in get_space(parametrizer)]
	symbols = [symbol for parametrizer in parametrizers for symbol in parametrizer.symbols]
	return Parametrizer(space, symbols)
end

function setindex!(parametrizer::Parametrizer, new_index::Vector{Int})
	index_type_accum = 1
	for single in parametrizer.single_types
		this_length = length(single)
		this_index = new_index[index_type_accum:index_type_accum+this_length-1]
		setindex!(single, this_index)
		new_index[index_type_accum:index_type_accum+this_length-1] = single.index
		index_type_accum = index_type_accum + this_length
	end
	parametrizer.index = new_index
end

function TEST_Parametrizer()
	test_Parametrizer = Parametrizer([["1", "2", "3"], ["1", "2", "3"], [:a3, :a45], [1, 2], [() -> 1.0, () -> 2.0]], [:I11, :I21, :I31, :I41, :I51])
	@assert length(test_Parametrizer.index) == length(values(test_Parametrizer)) == length(get_space(test_Parametrizer)) == length(test_Parametrizer)
	test_Parametrizer = Parametrizer([["1", "2", "3"], ["1", "2", "3"], [:a3, :a45], [1, 2], FunctionParameter.([() -> 1.0, () -> 2.0])], [:I11, :I21, :I31, :I41, :I51])
	@assert length(test_Parametrizer.index) == length(values(test_Parametrizer)) == length(get_space(test_Parametrizer)) == length(test_Parametrizer)
	test_Parametrizer_2 = deepcopy(test_Parametrizer)
	setindex!(test_Parametrizer_2, test_Parametrizer.index + ones(Int, length(test_Parametrizer)))
	setindex!(test_Parametrizer, test_Parametrizer.index + ones(Int, length(test_Parametrizer)))
	@assert test_Parametrizer_2.index == test_Parametrizer.index
	@assert values(test_Parametrizer) == values(get_space(test_Parametrizer), test_Parametrizer.index)

	test_Parametrizer_2 = Parametrizer([["1", "2", "3"], ["1", "2", "3"], [:a3, :a45], [1, 2], FunctionParameter.([() -> 1.0, () -> 2.0])], [:I12, :I22, :I32, :I42, :I52])
	test_parametrizer_3 = Parametrizer([test_Parametrizer, test_Parametrizer_2])
	println(test_parametrizer_3)

	try
		Parametrizer([["1", "2", "3"], ["1", "2", "3"], [:a3, :a45], [1, 2], FunctionParameter.([() -> 1.0, () -> 2.0])], [:I1, :I2, :I3, :I4])
		@assert false
	catch err
		@assert err isa ArgumentError
		@assert err.msg == "the space vector and symbols vector must be the same length."
	end
	return true
end

if TESTS
	@assert TEST_Parametrizer()
end

#end # module Parametrizer
