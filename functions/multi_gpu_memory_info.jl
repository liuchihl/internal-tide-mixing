function multi_gpu_memory_info()
   
    memory_info = []
    
    # Check all available GPUs
    num_devices = CUDA.ndevices()
    if num_devices > 0
        for gpu_id in 0:num_devices-1
            # Activate the specific GPU
            CUDA.device!(gpu_id)
            
            # Get memory information
            total_mem = CUDA.total_memory()
            free_mem = CUDA.available_memory()
            used_mem = total_mem - free_mem
            
            # Convert to GB for readability
            total_mem_gb = total_mem / (1024^3)
            used_mem_gb = used_mem / (1024^3)
            free_mem_gb = free_mem / (1024^3)
            
            # Create a formatted string for this GPU
            gpu_info = @sprintf("GPU %d: Total: %.2f GB, Used: %.2f GB, Free: %.2f GB (%.2f%%)", 
                                gpu_id, 
                                total_mem_gb, 
                                used_mem_gb, 
                                free_mem_gb, 
                                (used_mem / total_mem) * 100)
            
            push!(memory_info, gpu_info)
        end
    end
    
    return memory_info
end