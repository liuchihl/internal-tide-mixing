function mmintgrl(x::AbstractVector, y::Union{Vector, Matrix})
    # Ensure x is a column vector
    x = vec(x)
    nx = length(x)
    
    # Handle different input formats for y
    if size(y, 1) == 1 && size(y, 2) == nx
        y = transpose(y)
    end
    
    # Check dimensions
    if size(y, 1) != nx
        error("X and Y not the right size")
    end
    
    # Compute width of each trapezoid
    dx = x[2:end] - x[1:end-1]
    
    # Compute average of heights
    yave = (y[2:end, :] + y[1:end-1, :]) / 2
    
    # Multiply trapezoid widths by average heights
    area = dx .* yave
    
    # Cumulative sum along first dimension
    z = vcat(zeros(1, size(y, 2)), cumsum(area, dims=1))
    
    # Return in original orientation
    return size(y, 1) == 1 ? transpose(z) : z
end