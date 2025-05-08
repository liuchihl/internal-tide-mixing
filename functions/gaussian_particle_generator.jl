using CUDA
using Random

function gaussian_particle_generator(
    Nparticles, Lx, Nx, Ly, Ny, z_interp, architecture, H, θ;
    x_center_ratio=0.25, y_center_ratio=0.5, z_center=1000, σ_x=1000, σ_y=1000, σ_z=100
)
    is_3D = !isnothing(Ly) && !isnothing(Ny)

    # Define centers in Cartesian frame
    x_center = Lx * x_center_ratio  # Horizontal in Cartesian
    y_center = is_3D ? Ly * y_center_ratio : 0  # Unchanged
    z_center = z_center  # Depth in Cartesian

    # Use H as the maximum z-value
    max_z = H

    # Generate particles with Gaussian distribution
    ArrayType = architecture == CPU() ? Array : CuArray
    
    Nparticles = Int(Nparticles)
    if architecture == CPU()
        if is_3D
            samples = randn(Float64, (3, Nparticles))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= y_center .+ σ_y .* samples[2, :]
            samples[3, :] .= z_center .+ σ_z .* samples[3, :]
        else # for 2D
            samples = randn(Float64, (2, Nparticles))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= z_center .+ σ_z .* samples[2, :]
        end
        valid_mask = trues(Nparticles)
    else
        # For GPU, use CUDA's random number generation
        if is_3D
            samples = CUDA.randn(Float64, (3, Nparticles))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= y_center .+ σ_y .* samples[2, :]
            samples[3, :] .= z_center .+ σ_z .* samples[3, :]
        else # for 2D
            samples = CUDA.randn(Float64, (2, Nparticles))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= z_center .+ σ_z .* samples[2, :]
        end
        valid_mask = CUDA.ones(Bool, Nparticles)
    end

    # Transform to tilted frame (x', y', z')
    if is_3D
        x_prime = samples[1, :] .* cos(θ) - samples[3, :] .* sin(θ)
        y_prime = samples[2, :]  # y' = y
        z_prime = samples[1, :] .* sin(θ) + samples[3, :] .* cos(θ)
    else
        x_prime = samples[1, :] .* cos(θ) - samples[2, :] .* sin(θ)
        y_prime = zeros(Float64, Nparticles)  # y' = 0 in 2D
        z_prime = samples[1, :] .* sin(θ) + samples[2, :] .* cos(θ)
    end

    # Precompute values for GPU kernel
    x_scale = Float64(Nx / Lx)
    y_scale = is_3D ? Float64(Ny / Ly) : Float64(0)

    # Validate particles: some particles may be outside the domain or below the topography and we want to exclude them
    if architecture == CPU()
        for i in 1:Nparticles
            x = x_prime[i]
            z = z_prime[i]            
            x_in_bounds = 0 <= x <= Lx
            
            if x_in_bounds
                x_idx = max(1, min(Nx, round(Int, x * x_scale)))
                if is_3D
                    y = y_prime[i]
                    y_in_bounds = 0 <= y <= Ly
                    if y_in_bounds
                        y_idx = max(1, min(Ny, round(Int, y * y_scale)))
                        topo_z = z_interp[x_idx, y_idx]
                    else
                        valid_mask[i] = false
                        continue
                    end
                else
                    topo_z = z_interp[x_idx, 1]
                end
                
                if !(z > topo_z && z < max_z)
                    valid_mask[i] = false
                end
            else
                valid_mask[i] = false
            end
        end
    else
        # For GPU
        function validate_particle(x_prime, y_prime, z_prime, valid_mask, Lx, Nx, Ly, Ny, z_interp, max_z, is_3D, x_scale, y_scale)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(valid_mask)
                x = x_prime[i]
                z = z_prime[i]
                x_in_bounds = 0 <= x <= Lx
                if x_in_bounds
                    x_idx = max(1, min(Nx, round(Int32, x * x_scale)))
                    if is_3D
                        y = y_prime[i]
                        y_in_bounds = 0 <= y <= Ly
                        if y_in_bounds
                            y_idx = max(1, min(Ny, round(Int32, y * y_scale)))
                            topo_z = z_interp[x_idx, y_idx]
                        else
                            valid_mask[i] = false
                            return nothing
                        end
                    else
                        topo_z = z_interp[x_idx, 1]
                    end
                    if !(z > topo_z && z < max_z)
                        valid_mask[i] = false
                    end
                else
                    valid_mask[i] = false
                end
            end
            return nothing
        end

        x_prime_gpu = CuArray(x_prime)
        y_prime_gpu = CuArray(y_prime)
        z_prime_gpu = CuArray(z_prime)
        z_interp_gpu = CuArray(z_interp)
        threads = 256
        blocks = cld(Nparticles, threads)
        @cuda blocks=blocks threads=threads validate_particle(x_prime_gpu, y_prime_gpu, z_prime_gpu, valid_mask, Lx, Nx, Ly, Ny, z_interp_gpu, max_z, is_3D, x_scale, y_scale)
    end

    # Select valid particles
    valid_indices = findall(valid_mask)
    if length(valid_indices) < Nparticles
        @warn "Only generated $(length(valid_indices)) valid particles out of $(Nparticles) requested."
    end
    valid_indices = valid_indices[1:min(Nparticles, length(valid_indices))]

    # Return coordinates in tilted frame
    if is_3D
        x₀ = x_prime[valid_indices]
        y₀ = y_prime[valid_indices]
        z₀ = z_prime[valid_indices]
    else
        x₀ = x_prime[valid_indices]
        y₀ = architecture == CPU() ? zeros(Float64, length(valid_indices)) : CUDA.zeros(Float64, length(valid_indices))
        z₀ = z_prime[valid_indices]
    end

    return x₀, y₀, z₀
end