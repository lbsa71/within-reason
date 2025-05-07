"""
Simple script to check GPU status and reset CUDA context
"""

import torch
import gc
import os
import time
import psutil

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        print(f"\n===== GPU Memory Status {label} =====")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            print(f"  Reserved memory: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
            print(f"  Cached memory: {torch.cuda.memory_cached(i) / (1024**3):.2f} GB")
        print(f"Process memory: {get_process_memory():.2f} MB")
    else:
        print("CUDA not available")

def reset_gpu():
    print("\n===== Resetting GPU Memory =====")
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        # Create and delete a small tensor to reset CUDA context
        try:
            print("Creating test tensor...")
            x = torch.ones(1).cuda()
            del x
            print("Test tensor created and deleted successfully")
        except Exception as e:
            print(f"Error creating test tensor: {e}")
    
    print("Memory reset complete")

def run_simple_benchmark():
    """Run a simple benchmark to test GPU performance"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    print("\n===== Running Simple Benchmark =====")
    
    # Create a moderately sized tensor
    size = 5000
    print(f"Creating {size}x{size} matrices...")
    
    # Warm up
    torch.cuda.synchronize()
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    torch.cuda.synchronize()
    
    # Benchmark matrix multiplication
    print("Benchmarking matrix multiplication...")
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Matrix multiplication took {end - start:.4f} seconds")
    
    # Clean up
    del a, b, c
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print_gpu_memory("Before Reset")
    reset_gpu()
    print_gpu_memory("After Reset")
    run_simple_benchmark()
    print_gpu_memory("After Benchmark")
