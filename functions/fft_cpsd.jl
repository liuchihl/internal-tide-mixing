using FFTW, DSP

"""
    fft_cpsd(x, y, dt=1, win="rec")

1-sided sample estimate of the cross power spectral density (PSD)

This function uses FFT routine to calculate F.T and get the 
cross power spectral density 

# Arguments
- `x, y`: time series 
- `dt`: time interval (default: 1)
- `win`: window type - "bartlett", "hann", or "rec" (default: "rec")

# Returns
- `Sxx`: Auto power spectral density of x
- `Syy`: Auto power spectral density of y  
- `Sxy`: Cross power spectral density
- `fj`: Fourier frequencies

Written by Chih-Lun Liu, translated to Julia
Data Analysis OC683 (reference to Math Camp's note CHP26)
Original Date: May, 2021
"""
function fft_cpsd(x::Vector, y::Vector, dt::Real=1, win::String="rec")
    N = length(y)  # the observation record length
    
    # Generate frequencies
    if N % 2 == 0  # if N is even
        fj = collect(-N÷2:(N÷2-1)) / (N * dt)
    else  # if N is odd
        fj = collect(-(N-1)÷2:(N-1)÷2) / (N * dt)
    end
    
    # Apply windows
    x_windowed = copy(x)
    y_windowed = copy(y)
    
    if lowercase(win) == "bartlett"
        # Bartlett window (no need to boost for cross spectrum)
        window = bartlett(N)
        y_windowed .*= window
        x_windowed .*= window
    elseif lowercase(win) == "hann"
        # Hanning window
        window = hanning(N)
        y_windowed .*= window
        x_windowed .*= window
    else  # win == "rec"
        # Rectangle window - multiply by ones (no change)
        y_windowed = y_windowed .* ones(N)
        x_windowed = x_windowed .* ones(N)
    end
    
    # Compute auto and cross PSD (2-sided)
    X = fft(x_windowed, length(fj)) / N        # FT of x(n)
    X = fftshift(X)                            # shifted X
    Y = fft(y_windowed, length(fj)) / N        # FT of y(n)
    Y = fftshift(Y)                            # shifted Y
    
    # Auto PSD
    Sxx = X .* conj.(X) .* N .* dt        # power spectral density (2-sided)
    Syy = Y .* conj.(Y) .* N .* dt        # power spectral density (2-sided)
    
    # Cross PSD
    Sxy = conj.(X) .* Y .* N .* dt        # power spectral density (2-sided)
    
    # Convert PSD to 1-sided: S_prime = 2*S_hat, excluding fj=0
    positive_freq_mask = fj .> 0
    Sxx = 2 * Sxx[positive_freq_mask]
    Syy = 2 * Syy[positive_freq_mask]
    Sxy = Sxy[positive_freq_mask]    # no S prime for the cross spectrum,
                                     # so we have to multiply the coherence square
                                     # by 4 in the script
    fj = fj[positive_freq_mask]
    
    return Sxx, Syy, Sxy, fj
end