"""
    bin_stat_over_xy(var, bin_edge, bin_var; stat="mean")

Bin `var` by `bin_var` along the vertical, pooling over all horizontal locations, and compute the specified statistic for each bin (and optional time).

Supported statistics: "mean", "maximum", "minimum", "std", "median", "count".

Accepted shapes (var and bin_var must have the same shape):
- (nx, nz)           -> returns (nbins)
- (nx, ny, nz)       -> returns (nbins)
- (nz, nt)           -> returns (nbins, nt)
- (nx, ny, nz, nt)   -> returns (nbins, nt)

Note: The function treats the last dimension as time if present. All horizontal dimensions (x[, y]) are pooled within each vertical bin.
"""
function bin_stat_over_xy(var, bin_edge, bin_var; stat="mean")
    # bring in common stats (mean/std) explicitly
    # (Base.maximum/minimum are already available)
    sz = size(var)
    nd = ndims(var)
    nbins = length(bin_edge) - 1

    # Basic validation
    size(var) == size(bin_var) || error("var and bin_var must have the same size; got $(size(var)) vs $(size(bin_var))")
    nbins > 0 || error("bin_edge must contain at least two values (>= 1 bin)")

    # Statistics dispatcher
    stat_fun = Dict(
        "mean"    => x -> isempty(x) ? NaN : mean(x),
        "maximum" => x -> isempty(x) ? NaN : maximum(x),
        "minimum" => x -> isempty(x) ? NaN : minimum(x),
        "std"     => x -> isempty(x) ? NaN : std(x),
        "median"  => x -> isempty(x) ? NaN : median(x),
        "count"   => x -> length(x)
    )
    haskey(stat_fun, stat) || error("Statistic '$stat' not supported. Choose from: $(collect(keys(stat_fun)))")

    # Helper to compute bin index (0 if out of range)
    bin_index(v) = (searchsortedfirst(bin_edge, v) - 1)

    # 1) 4-D: (nx, ny, nz, nt)
    if nd == 4
        nx, ny, nz, nt = sz
        result = Array{Float64}(undef, nbins, nt)
        @inbounds for it in 1:nt
            # one accumulator per bin
            vals = [Float64[] for _ in 1:nbins]
            for ix in 1:nx, iy in 1:ny, iz in 1:nz
                val = var[ix, iy, iz, it]
                if !isnan(val)
                    b = bin_index(bin_var[ix, iy, iz, it])
                    if 1 <= b <= nbins
                        push!(vals[b], float(val))
                    end
                end
            end
            for k in 1:nbins
                result[k, it] = stat_fun[stat](vals[k])
            end
        end
        return result
    end

    # 2) 3-D: (nx, ny, nz)
    if nd == 3
        nx, ny, nz = sz
        result = Array{Float64}(undef, nbins)
        vals = [Float64[] for _ in 1:nbins]
        @inbounds for ix in 1:nx, iy in 1:ny, iz in 1:nz
            val = var[ix, iy, iz]
            if !isnan(val)
                b = bin_index(bin_var[ix, iy, iz])
                if 1 <= b <= nbins
                    push!(vals[b], float(val))
                end
            end
        end
        for k in 1:nbins
            result[k] = stat_fun[stat](vals[k])
        end
        return result
    end

    # 3) 2-D: could be (nx, nz) or (nz, nt).
    # We interpret the last dimension as time (if it is "time-like").
    # Heuristic: if either dimension equals 1 -> handle naturally; else, if
    # bin_var varies along the second dim (likely time), treat as (nz, nt).
    if nd == 2
        n1, n2 = sz

        # Try to detect (nz, nt) pattern: vertical coordinate typically varies in the first dim
        # If the second dimension seems like time (bin_var columns share similar range), we handle (nz, nt).
        # Otherwise, default to (nx, nz).
        treat_as_nz_nt = false
        if n1 > 1 && n2 > 1
            # If bin_var has identical first-column and last-column lengths and ranges are similar,
            # assume (nz, nt). This is a light heuristic that should work when inputs are (nz, nt).
            r1 = (minimum(bin_var[:, 1]), maximum(bin_var[:, 1]))
            r2 = (minimum(bin_var[:, end]), maximum(bin_var[:, end]))
            treat_as_nz_nt = (r1[1] <= r2[2] && r2[1] <= r1[2]) # overlapping ranges
        end

        if treat_as_nz_nt
            nz, nt = n1, n2
            result = Array{Float64}(undef, nbins, nt)
            @inbounds for it in 1:nt
                vals = [Float64[] for _ in 1:nbins]
                for iz in 1:nz
                    val = var[iz, it]
                    if !isnan(val)
                        b = bin_index(bin_var[iz, it])
                        if 1 <= b <= nbins
                            push!(vals[b], float(val))
                        end
                    end
                end
                for k in 1:nbins
                    result[k, it] = stat_fun[stat](vals[k])
                end
            end
            return result
        else
            # Treat as (nx, nz): pool over x and z into vertical bins
            nx, nz = n1, n2
            result = Array{Float64}(undef, nbins)
            vals = [Float64[] for _ in 1:nbins]
            @inbounds for ix in 1:nx, iz in 1:nz
                val = var[ix, iz]
                if !isnan(val)
                    b = bin_index(bin_var[ix, iz])
                    if 1 <= b <= nbins
                        push!(vals[b], float(val))
                    end
                end
            end
            for k in 1:nbins
                result[k] = stat_fun[stat](vals[k])
            end
            return result
        end
    end

    # 4) 1-D: (nz)
    if nd == 1
        nz = sz[1]
        result = Array{Float64}(undef, nbins)
        vals = [Float64[] for _ in 1:nbins]
        @inbounds for iz in 1:nz
            val = var[iz]
            if !isnan(val)
                b = bin_index(bin_var[iz])
                if 1 <= b <= nbins
                    push!(vals[b], float(val))
                end
            end
        end
        for k in 1:nbins
            result[k] = stat_fun[stat](vals[k])
        end
        return result
    end

    error("Unsupported array dimensionality: ndims(var)=$(nd). Supported: 1D, 2D, 3D, 4D.")
end

"""
    bin_stat_over_xy_overlapping(var, bin_centers, halfwidth, bin_var; stat="mean")

Compute statistics in overlapping vertical bins defined by centers and half-width.

Each bin k includes all points whose bin_var satisfies:
    bin_centers[k] - halfwidth ≤ bin_var < bin_centers[k] + halfwidth

Shapes supported are identical to `bin_stat_over_xy` (1D, 2D, 3D, 4D), with time optionally in the last dimension.
Returns (nbins) or (nbins, nt) accordingly.
"""
function bin_stat_over_xy_overlapping(var, bin_centers, halfwidth, bin_var; stat="mean")
    size(var) == size(bin_var) || error("var and bin_var must have the same size; got $(size(var)) vs $(size(bin_var))")
    nbins = length(bin_centers)
    nbins > 0 || error("bin_centers must be non-empty")
    halfwidth > 0 || error("halfwidth must be positive")

    stat_fun = Dict(
        "mean"    => x -> isempty(x) ? NaN : mean(x),
        "maximum" => x -> isempty(x) ? NaN : maximum(x),
        "minimum" => x -> isempty(x) ? NaN : minimum(x),
        "std"     => x -> isempty(x) ? NaN : std(x),
        "median"  => x -> isempty(x) ? NaN : median(x),
        "count"   => x -> length(x)
    )
    haskey(stat_fun, stat) || error("Statistic '$stat' not supported. Choose from: $(collect(keys(stat_fun)))")

    nd = ndims(var)
    sz = size(var)

    # Helper: bin membership for a value
    @inline function bin_memberships(v)
        # Return indices of bins whose interval contains v
        # Using vectorized check; for performance you could precompute bounds
        idxs = Int[]
        @inbounds for k in 1:nbins
            if (bin_centers[k] - halfwidth) <= v < (bin_centers[k] + halfwidth)
                push!(idxs, k)
            end
        end
        return idxs
    end

    if nd == 4
        nx, ny, nz, nt = sz
        result = Array{Float64}(undef, nbins, nt)
        @inbounds for it in 1:nt
            vals = [Float64[] for _ in 1:nbins]
            for ix in 1:nx, iy in 1:ny, iz in 1:nz
                val = var[ix, iy, iz, it]
                if !isnan(val)
                    idxs = bin_memberships(bin_var[ix, iy, iz, it])
                    for k in idxs
                        push!(vals[k], float(val))
                    end
                end
            end
            for k in 1:nbins
                result[k, it] = stat_fun[stat](vals[k])
            end
        end
        return result
    elseif nd == 3
        nx, ny, nz = sz
        result = Array{Float64}(undef, nbins)
        vals = [Float64[] for _ in 1:nbins]
        @inbounds for ix in 1:nx, iy in 1:ny, iz in 1:nz
            val = var[ix, iy, iz]
            if !isnan(val)
                idxs = bin_memberships(bin_var[ix, iy, iz])
                for k in idxs
                    push!(vals[k], float(val))
                end
            end
        end
        for k in 1:nbins
            result[k] = stat_fun[stat](vals[k])
        end
        return result
    elseif nd == 2
        n1, n2 = sz
        # Heuristic: if treating as (nz, nt)
        # Use same logic as bin_stat_over_xy: check if bin_var ranges overlap across columns
        treat_as_nz_nt = false
        if n1 > 1 && n2 > 1
            r1 = (minimum(bin_var[:, 1]), maximum(bin_var[:, 1]))
            r2 = (minimum(bin_var[:, end]), maximum(bin_var[:, end]))
            treat_as_nz_nt = (r1[1] <= r2[2] && r2[1] <= r1[2]) # overlapping ranges suggest time
        end
        if treat_as_nz_nt
            nz, nt = n1, n2
            result = Array{Float64}(undef, nbins, nt)
            @inbounds for it in 1:nt
                vals = [Float64[] for _ in 1:nbins]
                for iz in 1:nz
                    val = var[iz, it]
                    if !isnan(val)
                        idxs = bin_memberships(bin_var[iz, it])
                        for k in idxs
                            push!(vals[k], float(val))
                        end
                    end
                end
                for k in 1:nbins
                    result[k, it] = stat_fun[stat](vals[k])
                end
            end
            return result
        else
            nx, nz = n1, n2
            result = Array{Float64}(undef, nbins)
            vals = [Float64[] for _ in 1:nbins]
            @inbounds for ix in 1:nx, iz in 1:nz
                val = var[ix, iz]
                if !isnan(val)
                    idxs = bin_memberships(bin_var[ix, iz])
                    for k in idxs
                        push!(vals[k], float(val))
                    end
                end
            end
            for k in 1:nbins
                result[k] = stat_fun[stat](vals[k])
            end
            return result
        end
    elseif nd == 1
        nz = sz[1]
        result = Array{Float64}(undef, nbins)
        vals = [Float64[] for _ in 1:nbins]
        @inbounds for iz in 1:nz
            val = var[iz]
            if !isnan(val)
                idxs = bin_memberships(bin_var[iz])
                for k in idxs
                    push!(vals[k], float(val))
                end
            end
        end
        for k in 1:nbins
            result[k] = stat_fun[stat](vals[k])
        end
        return result
    else
        error("Unsupported array dimensionality: ndims(var)=$(nd). Supported: 1D, 2D, 3D, 4D.")
    end
end

"""
    bin_stat_over_xy_overlapping(var, start, stop, width, stride, bin_var; stat="mean")

Convenience overload to compute overlapping-bin statistics by specifying the interval parameters.

Parameters:
- start, stop: bounds for bin centers (inclusive of start, <= stop)
- width: total bin width (e.g., 8 means ±4 halfwidth)
- stride: spacing between successive bin centers (e.g., 4)

Equivalent to calling `bin_stat_over_xy_overlapping(var, start:stride:stop, width/2, bin_var; stat=stat)`.
"""
function bin_stat_over_xy_overlapping(var, start, stop, width, stride, bin_var; stat="mean")
    width > 0 || error("width must be positive")
    stride > 0 || error("stride must be positive")
    start <= stop || error("start must be <= stop")
    centers = start:stride:stop
    return bin_stat_over_xy_overlapping(var, centers, width/2, bin_var; stat=stat)
end