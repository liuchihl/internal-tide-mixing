using Oceananigans, Oceananigans.Fields
using Oceananigans.AbstractOperations: ConditionalOperation
using LinearAlgebra, CUDA

# Override the problematic sum implementation for ConditionalOperation
function Base.sum(f::Function, 
                 c::ConditionalOperation; 
                 condition=nothing, 
                 mask=0,
                 dims=:)
    if dims isa Colon
        raw_data = parent(interior(c))
        return sum(f.(raw_data))
    else
        raw_data = parent(interior(c))
        return sum(f.(raw_data), dims=dims)
    end
end

# Override the norm function for AbstractField
function LinearAlgebra.norm(a::AbstractField; condition = nothing)
    raw_data = parent(interior(condition_operand(a, condition, 0)))
    squared_sum = sum(x -> x*x, raw_data)
    return sqrt(squared_sum)
end
