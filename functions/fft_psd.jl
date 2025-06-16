using FFTW

# 1-sided sample estimate of the power spectral density (PSD)
function fft_psd(y; dt=1, win="rec", Np=nothing)
    N = length(y)                  
    
    # Apply window first
    if lowercase(win) == "bartlett"
        var_y = mean(y.^2) - mean(y)^2    
        bartlett_win = 1 .- 2 .* abs.((0:N-1) .- (N-1)/2) ./ (N-1)
        y = y .* bartlett_win             
        var_yr = mean(y.^2) - mean(y)^2   
        boost_factor = sqrt(var_y/var_yr)
    elseif lowercase(win) == "hann"
        hann_win = 0.5 .* (1 .- cos.(2π .* (0:N-1) ./ (N-1)))
        y = y .* hann_win
    else  # win == "rec"
        y = y .* ones(N)
    end
    
    # Perform FFT
    if isnothing(Np)
        # Calculate frequencies first
        if N % 2 == 0                      
            fj = (-N÷2:N÷2-1) ./ (N*dt)    
        else                              
            fj = (-(N-1)÷2:(N-1)÷2) ./ (N*dt)
        end
        
        # Calculate FFT
        Y = fft(y)/N                      # Just use fft(y) without the second arg
        PSD = abs2.(Y) .* N .* dt         # Use abs2 instead of Y .* conj(Y)
        PSD = fftshift(PSD)               
    else
        # Zero padding case
        fj = (-Np÷2:Np÷2-1) ./ (Np*dt)    
        Y = fft(y, Np)/Np                 # Use Np directly
        PSD = abs2.(Y) .* N .* dt        
        PSD = fftshift(PSD)              
    end
    
    # Convert to 1-sided PSD
    positive_indices = findall(fj .> 0)
    fj = fj[positive_indices]
    Sp = 2 .* PSD[positive_indices]
    
    # Apply boost if Bartlett window was used
    if lowercase(win) == "bartlett"
        Sp = Sp .* boost_factor^2
    end
    
    return Sp, fj
end