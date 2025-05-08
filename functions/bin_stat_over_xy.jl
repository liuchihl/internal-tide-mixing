"""
    bin_stat_over_xy(var, bin_edge, bin_var; stat="mean")

Bin `var` by `bin_var` along the vertical, pooling over all (x, y) locations, and compute the specified statistic for each bin (and [t]).
Supported statistics: "mean", "maximum", "minimum", "std", "median", "count".
Returns an array of size (nbins, [nt]).
"""
function bin_stat_over_xy(var, bin_edge, bin_var; stat="mean")
    sz = size(var)
    nbins = length(bin_edge) - 1
    stat_fun = Dict(
        "mean"    => x -> isempty(x) ? NaN : mean(x),
        "maximum" => x -> isempty(x) ? NaN : maximum(x),
        "minimum" => x -> isempty(x) ? NaN : minimum(x),
        "std"     => x -> isempty(x) ? NaN : std(x),
        "median"  => x -> isempty(x) ? NaN : median(x),
        "count"   => x -> length(x)
    )
    if !haskey(stat_fun, stat)
        error("Statistic '$stat' not supported. Choose from: $(collect(keys(stat_fun)))")
    end

    if length(sz) == 3
        vals = [[] for _ in 1:nbins]
        nx, ny, nz = sz
        for ix in 1:nx, iy in 1:ny, iz in 1:nz
            val = var[ix, iy, iz]
            binval = bin_var[ix, iy, iz]
            bin_idx = searchsortedfirst(bin_edge, binval) - 1
            if bin_idx >= 1 && bin_idx <= nbins && !isnan(val)
                push!(vals[bin_idx], val)
            end
        end
        result = [stat_fun[stat](vals[k]) for k in 1:nbins]
    else
        nx, ny, nz, nt = sz
        result = Array{Float64}(undef, nbins, nt)
        for it in 1:nt
            vals = [[] for _ in 1:nbins]
            for ix in 1:nx, iy in 1:ny, iz in 1:nz
                val = var[ix, iy, iz, it]
                binval = bin_var[ix, iy, iz, it]
                bin_idx = searchsortedfirst(bin_edge, binval) - 1
                if bin_idx >= 1 && bin_idx <= nbins && !isnan(val)
                    push!(vals[bin_idx], val)
                end
            end
            for k in 1:nbins
                result[k, it] = stat_fun[stat](vals[k])
            end
        end
    end
    return result
end