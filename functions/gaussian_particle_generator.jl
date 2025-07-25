using CUDA
using Random

function gaussian_particle_generator(
    Nparticles, Lx, Nx, Ly, Ny, z_interp, architecture, H;
    x_center_ratio=0.25, y_center_ratio=0.5, z_center=1000, σ_x=1000, σ_y=1000, σ_z=100,
    apply_periodic_bounds=true
)
# apply_periodic_bounds: if true, particles can wrap around the domain in x and y directions
# this happens when picking up from a checkpoint file, where particles may be outside the domain from previous run
    is_3D = !isnothing(Ly) && !isnothing(Ny)

    x_center = Lx * x_center_ratio
    if is_3D
        y_center = Ly * y_center_ratio
    else
        y_center = 0
    end

    # set z_center: the depth of the center patch
    z_center = z_center

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

    # Precompute values for GPU kernel (only needed for validation)
    x_scale = Float64(Nx / Lx)
    y_scale = is_3D ? Float64(Ny / Ly) : Float64(0)

    # Handle particle validation based on boundary conditions
    if apply_periodic_bounds
        # When using periodic bounds, only apply wrapping - no validation
        # Trust immersed boundaries to handle topography interactions
        if architecture == CPU()
            for i in 1:Nparticles
                # Apply periodic wrapping for horizontal coordinates
                samples[1, i] = mod(samples[1, i], Lx)
                if is_3D
                    samples[2, i] = mod(samples[2, i], Ly)
                end
                # No vertical validation - immersed boundary handles this
            end
        else
            # GPU version for periodic wrapping
            function wrap_particles_gpu(sample, Lx, Ly, is_3D)
                i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                if i <= size(sample, 2)
                    sample[1, i] = mod(sample[1, i], Lx)
                    if is_3D
                        sample[2, i] = mod(sample[2, i], Ly)
                    end
                end
                return nothing
            end

            threads = 256
            blocks = cld(Nparticles, threads)
            @cuda blocks=blocks threads=threads wrap_particles_gpu(samples, Lx, Ly, is_3D)
        end
        
        # All particles are valid when using periodic bounds
        valid_indices = 1:Nparticles
        
    else
        # Original validation logic for initial generation (non-periodic)
        if architecture == CPU()
            for i in 1:Nparticles
                x = samples[1, i]
                z = is_3D ? samples[3, i] : samples[2, i]
                
                x_in_bounds = (0 <= x <= Lx)
                
                if x_in_bounds
                    x_idx = max(1, min(Nx, round(Int, x * x_scale)))
                    if is_3D
                        y = samples[2, i]
                        y_in_bounds = (0 <= y <= Ly)
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
                    
                    # Validate vertical bounds for initial generation
                    if !(z > topo_z && z < max_z)
                        valid_mask[i] = false
                    end
                else
                    valid_mask[i] = false
                end
            end
        else
            # GPU validation for initial generation
            function validate_particle_initial(sample, valid_mask, Lx, Nx, Ly, Ny, z_interp, max_z, is_3D, x_scale, y_scale)
                i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                if i <= length(valid_mask)
                    x = sample[1, i]
                    z = is_3D ? sample[3, i] : sample[2, i]
                    
                    x_in_bounds = (0 <= x <= Lx)
                    
                    if x_in_bounds
                        x_idx = max(1, min(Nx, round(Int32, x * x_scale)))
                        if is_3D
                            y = sample[2, i]
                            y_in_bounds = (0 <= y <= Ly)
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
                        
                        # Validate vertical bounds
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
            blocks = cld(Nparticles, threads)
            @cuda blocks=blocks threads=threads validate_particle_initial(samples, valid_mask, Lx, Nx, Ly, Ny, z_interp_gpu, max_z, is_3D, x_scale, y_scale)
        end
        
        # Select valid particles for initial generation
        valid_indices = findall(valid_mask)
        if length(valid_indices) < Nparticles
            @warn "Only generated $(length(valid_indices)) valid particles out of $(Nparticles) requested."
        end
        valid_indices = valid_indices[1:min(Nparticles, length(valid_indices))]
    end

    # Extract particle positions
    if is_3D
        x₀ = samples[1, valid_indices]
        y₀ = samples[2, valid_indices]
        z₀ = samples[3, valid_indices]
    else
        x₀ = samples[1, valid_indices]
        y₀ = architecture == CPU() ? zeros(Float64, length(valid_indices)) : CUDA.zeros(Float64, length(valid_indices))
        z₀ = samples[2, valid_indices]
    end

    return x₀, y₀, z₀
end