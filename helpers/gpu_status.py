import pynvml


pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

def get_gpu_status():
    """
    Retrieve GPU utilization and memory usage using pynvml.
    """
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return {
        "gpu_utilization": util.gpu,  # GPU usage percentage
        "memory_used_mb": mem_info.used / 1024 ** 2,  # Memory used in MB
        "memory_total_mb": mem_info.total / 1024 ** 2  # Total memory in MB
    }