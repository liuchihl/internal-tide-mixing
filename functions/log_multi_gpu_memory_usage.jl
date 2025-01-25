function log_multi_gpu_memory_usage()
    # Initialize an empty string to store the status of all GPUs
    status = ""
    
    # Get the number of visible CUDA devices
    num_devices = CUDA.ndevices()
    total_memory = 0
    
    for device in 0:(num_devices-1)
        CUDA.device!(device)  # Switch to this device
        mem_info = CUDA.memory_status()
        total_memory += CUDA.total_memory()  # Add to total memory capacity
        
        # Append this device's status to our string
        status *= "GPU $device: $mem_info; "
    end
    
    # Add total capacity information
    total_memory_gb = total_memory / 1024^3  # Convert to GB
    status *= @sprintf("Total GPU Memory Capacity: %.2f GB", total_memory_gb)
    
    return status
end