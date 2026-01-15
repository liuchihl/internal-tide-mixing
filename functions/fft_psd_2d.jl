using FFTW

"""
    fft_psd_2d_onesided(data; dx=1.0, dt=1.0, win_space="rec", win_time="rec")

Compute 1-sided 2D power spectral density of 2D data [x, time].
Windowing can be applied independently in x (wavenumber) and t (frequency) directions.
Returns only positive kx and freq, with proper scaling.
"""
function fft_psd_2d(data; dx=1.0, dt=1.0, win_space="rec", win_time="rec")
    Nx, Nt = size(data)
    # Windowing in x
    if lowercase(win_space) == "bartlett"
        wx = 1 .- 2 .* abs.((0:Nx-1) .- (Nx-1)/2) ./ (Nx-1)
    elseif lowercase(win_space) == "hann"
        wx = 0.5 .* (1 .- cos.(2π .* (0:Nx-1) ./ (Nx-1)))
    else
        wx = ones(Nx)
    end
    # Windowing in t
    if lowercase(win_time) == "bartlett"
        wt = 1 .- 2 .* abs.((0:Nt-1) .- (Nt-1)/2) ./ (Nt-1)
    elseif lowercase(win_time) == "hann"
        wt = 0.5 .* (1 .- cos.(2π .* (0:Nt-1) ./ (Nt-1)))
    else
        wt = ones(Nt)
    end
    window_2d = wx * wt'
    windowed_data = data .* window_2d

    # 2D FFT
    Y = fft(windowed_data) / (Nx * Nt)
    PSD = abs2.(Y) .* Nx .* Nt .* dx .* dt

    # 1-sided indices for kx and freq
    kx_idx = 1:(Nx ÷ 2 + 1)
    freq_idx = 1:(Nt ÷ 2 + 1)
    kx = (0:(Nx ÷ 2)) ./ (Nx * dx)
    freq = (0:(Nt ÷ 2)) ./ (Nt * dt)

    # Extract 1-sided PSD
    PSD_1sided = PSD[kx_idx, freq_idx]

    # Double non-DC/non-Nyquist components to conserve power
    for i in 1:length(kx)
        for j in 1:length(freq)
            if (i != 1 && (Nx % 2 == 0 ? i != length(kx) : true)) &&
               (j != 1 && (Nt % 2 == 0 ? j != length(freq) : true))
                PSD_1sided[i, j] *= 2
            elseif (i != 1 && (Nx % 2 == 0 ? i != length(kx) : true))
                PSD_1sided[i, j] *= 2
            elseif (j != 1 && (Nt % 2 == 0 ? j != length(freq) : true))
                PSD_1sided[i, j] *= 2
            end
        end
    end

    return PSD_1sided, kx, freq
end