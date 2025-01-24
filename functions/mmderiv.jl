using LinearAlgebra
using Polynomials


function mmderiv(x, y)
    x = float.(x)  # Ensure x is floating-point to avoid integer division issues
    nx = length(x)
    
    if ndims(y) == 1
        y = reshape(y, :, 1)
        ry, cy = size(y)
        flag = true
    else
        ry, cy = size(y)
        flag = false
    end
    
    if nx != ry
        error("X and Y not the right size")
    end
    if nx < 3
        error("X and Y must have at least three elements")
    end

    dx = x[2:nx] - x[1:nx-1]
    dx[dx .== 0] .= eps()
    dxx = x[3:nx] - x[1:nx-2]
    dxx[dxx .== 0] .= eps()
    alpha = dx[1:nx-2] ./ dxx
    alpha = repeat(alpha, 1, cy)

    dy = y[2:ry, :] - y[1:ry-1, :]
    dx = repeat(dx, 1, cy)

    z = alpha .* dy[2:ry-1, :] ./ dx[2:nx-1, :] + (1 .- alpha) .* dy[1:ry-2, :] ./ dx[1:nx-2, :]

    z1 = zeros(1, cy)
    zn = z1
    for i = 1:cy
        p1 = Polynomials.fit(Polynomial, x[1:3], y[1:3, i], 2)
        z1[i] = derivative(p1)(x[1])
        pn = Polynomials.fit(Polynomial, x[nx-2:nx], y[ry-2:ry, i], 2)
        zn[i] = derivative(pn)(x[nx])
    end

    z = vcat(z1, z, zn)
    if flag
        z = z[:]
    end
    return z
end