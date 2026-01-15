using FFTW

"""
    fft_cpsd_2d(data_x, data_y; dx=1.0, dt=1.0, win="rec")

Compute 1-sided 2D cross power spectral density of 2D data [x, time].
Returns only positive kx and freq, with proper scaling.

# Arguments
- `data_x, data_y`: 2D time series data [spatial, time]
- `dx`: spatial interval (default: 1.0)
- `dt`: time interval (default: 1.0)
- `win`: window type - "bartlett", "hann", or "rec" (default: "rec")

# Returns
- `Sxx`: Auto power spectral density of data_x
- `Syy`: Auto power spectral density of data_y  
- `Sxy`: Cross power spectral density
- `kx`: Spatial wavenumbers (positive only)
- `freq`: Frequencies (positive only)

Based on 1D cross spectral code by Chih-Lun Liu, extended to 2D
"""
function fft_cpsd_2d(data_x, data_y; dx=1.0, dt=1.0, win="rec")
    Nx, Nt = size(data_x)
    
    # Check that both datasets have the same size
    if size(data_x) != size(data_y)
        error("data_x and data_y must have the same dimensions")
    end
    
    # Apply windowing
    windowed_x = copy(data_x)
    windowed_y = copy(data_y)
    
    if lowercase(win) == "bartlett"
        bartlett_x = 1 .- 2 .* abs.((0:Nx-1) .- (Nx-1)/2) ./ (Nx-1)
        bartlett_t = 1 .- 2 .* abs.((0:Nt-1) .- (Nt-1)/2) ./ (Nt-1)
        window_2d = bartlett_x * bartlett_t'
        windowed_x = data_x .* window_2d
        windowed_y = data_y .* window_2d
    elseif lowercase(win) == "hann"
        hann_x = 0.5 .* (1 .- cos.(2π .* (0:Nx-1) ./ (Nx-1)))
        hann_t = 0.5 .* (1 .- cos.(2π .* (0:Nt-1) ./ (Nt-1)))
        window_2d = hann_x * hann_t'
        windowed_x = data_x .* window_2d
        windowed_y = data_y .* window_2d
    else  # win == "rec"
        # Rectangle window - no change needed
        windowed_x = data_x
        windowed_y = data_y
    end
    
    # 2D FFT
    X = fft(windowed_x) / (Nx * Nt)    # FT of data_x
    Y = fft(windowed_y) / (Nx * Nt)    # FT of data_y
    
    # Compute auto and cross PSD (2-sided)
    Sxx = abs2.(X) .* Nx .* Nt .* dx .* dt     # Auto PSD of X
    Syy = abs2.(Y) .* Nx .* Nt .* dx .* dt     # Auto PSD of Y
    Sxy = conj.(X) .* Y .* Nx .* Nt .* dx .* dt   # Cross PSD
    
    # 1-sided indices for kx and freq
    kx_idx = 1:(Nx ÷ 2 + 1)
    freq_idx = 1:(Nt ÷ 2 + 1)
    kx = (0:(Nx ÷ 2)) ./ (Nx * dx)
    freq = (0:(Nt ÷ 2)) ./ (Nt * dt)
    
    # Extract 1-sided spectra
    Sxx_1sided = Sxx[kx_idx, freq_idx]
    Syy_1sided = Syy[kx_idx, freq_idx]
    Sxy_1sided = Sxy[kx_idx, freq_idx]
    
    # Double non-DC/non-Nyquist components to conserve power for auto spectra
    # Cross spectrum scaling is different - no doubling needed
    for i in 1:length(kx)
        for j in 1:length(freq)
            # Conditions for doubling: not DC (i≠1, j≠1) and not Nyquist
            double_kx = (i != 1 && (Nx % 2 == 0 ? i != length(kx) : true))
            double_freq = (j != 1 && (Nt % 2 == 0 ? j != length(freq) : true))
            
            if double_kx && double_freq
                # Both kx and freq are not DC/Nyquist - double auto spectra
                Sxx_1sided[i, j] *= 2
                Syy_1sided[i, j] *= 2
                # Cross spectrum: no doubling (similar to 1D case comment)
            elseif double_kx
                # Only kx is not DC/Nyquist
                Sxx_1sided[i, j] *= 2
                Syy_1sided[i, j] *= 2
            elseif double_freq
                # Only freq is not DC/Nyquist  
                Sxx_1sided[i, j] *= 2
                Syy_1sided[i, j] *= 2
            end
            # If both are DC or Nyquist, no scaling needed
        end
    end
    
    return Sxx_1sided, Syy_1sided, Sxy_1sided, kx, freq
end