using CUDA
using Random

function gaussian_particle_generator(
    Nparticles, Lx, Nx, Ly, Ny, z_interp, architecture, H;
    x_center_ratio=0.25, y_center_ratio=0.5, z_above_bottom=100, σ_x=1000, σ_y=1000, σ_z=100
)
    is_3D = !isnothing(Ly) && !isnothing(Ny)

    x_center = Lx * x_center_ratio
    if is_3D
        y_center = Ly * y_center_ratio
    else
        y_center = 0
    end

    # Find the z-center
    # Make sure indices are within valid bounds (1 to Nx/Ny)
    x_center_idx = max(1, min(Nx, round(Int, x_center / Lx * Nx)))
    if is_3D
        y_center_idx = max(1, min(Ny, round(Int, y_center / Ly * Ny)))
        bottom_z_at_center = z_interp[x_center_idx, y_center_idx]
    else
        bottom_z_at_center = z_interp[x_center_idx, 1]
    end
    z_center = bottom_z_at_center + z_above_bottom

    # Use H as the maximum z-value
    max_z = H

    # Generate particles with Gaussian distribution
    ArrayType = architecture == CPU() ? Array : CuArray
    
    # Generate more particles than needed to account for invalid ones
    oversampling_factor = 2
    n_samples = Nparticles * oversampling_factor

    if architecture == CPU()
        if is_3D
            samples = randn(Float32, (3, n_samples))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= y_center .+ σ_y .* samples[2, :]
            samples[3, :] .= z_center .+ σ_z .* samples[3, :]
        else # for 2D
            samples = randn(Float32, (2, n_samples))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= z_center .+ σ_z .* samples[2, :]
        end
        valid_mask = trues(n_samples)
    else
        # For GPU, use CUDA's random number generation
        if is_3D
            samples = CUDA.randn(Float32, (3, n_samples))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= y_center .+ σ_y .* samples[2, :]
            samples[3, :] .= z_center .+ σ_z .* samples[3, :]
        else # for 2D
            samples = CUDA.randn(Float32, (2, n_samples))
            samples[1, :] .= x_center .+ σ_x .* samples[1, :]
            samples[2, :] .= z_center .+ σ_z .* samples[2, :]
        end
        valid_mask = CUDA.ones(Bool, n_samples)
    end

    # Precompute values for GPU kernel
    x_scale = Float32(Nx / Lx)
    y_scale = is_3D ? Float32(Ny / Ly) : Float32(0)

    # Validate particles
    if architecture == CPU()
        for i in 1:n_samples
            x = samples[1, i]
            z = is_3D ? samples[3, i] : samples[2, i]
            
            x_in_bounds = 0 <= x <= Lx
            
            if x_in_bounds
                x_idx = max(1, min(Nx, round(Int, x * x_scale)))
                if is_3D
                    y = samples[2, i]
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
        # For GPU, we need to use a kernel or broadcast operation
        function validate_particle(sample, valid_mask, Lx, Nx, Ly, Ny, z_interp, max_z, is_3D, x_scale, y_scale)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(valid_mask)
                x = sample[1, i]
                z = is_3D ? sample[3, i] : sample[2, i]
                
                x_in_bounds = 0 <= x <= Lx
                
                if x_in_bounds
                    x_idx = max(1, min(Nx, round(Int32, x * x_scale)))
                    if is_3D
                        y = sample[2, i]
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

        # Convert z_interp to CuArray
        z_interp_gpu = CuArray(z_interp)

        threads = 256
        blocks = cld(n_samples, threads)
        @cuda blocks=blocks threads=threads validate_particle(samples, valid_mask, Lx, Nx, Ly, Ny, z_interp_gpu, max_z, is_3D, x_scale, y_scale)
    end

    # Select valid particles
    valid_indices = findall(valid_mask)
    if length(valid_indices) < Nparticles
        @warn "Only generated $(length(valid_indices)) valid particles out of $(Nparticles) requested."
    end
    valid_indices = valid_indices[1:min(Nparticles, length(valid_indices))]

    if is_3D
        x₀ = samples[1, valid_indices]
        y₀ = samples[2, valid_indices]
        z₀ = samples[3, valid_indices]
    else
        x₀ = samples[1, valid_indices]
        y₀ = architecture == CPU() ? zeros(Float32, length(valid_indices)) : CUDA.zeros(Float32, length(valid_indices))
        z₀ = samples[2, valid_indices]
    end

    return x₀, y₀, z₀
end