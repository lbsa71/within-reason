"""
Helper script to run benchmarks with different configurations.
"""

import os
import sys
import subprocess
import argparse
import torch
from datetime import datetime

def check_cuda_availability(device):
    """Check if CUDA is available when requested."""
    if device == "cuda":
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if not cuda_available:
            print("ERROR: CUDA device requested but CUDA is not available.")
            print("Your PyTorch installation does not have CUDA support.")
            print("\nTo fix this, reinstall PyTorch with CUDA support:")
            print("pip uninstall -y torch")
            print("pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("\nOr run the benchmark on CPU instead:")
            print("python run_benchmarks.py --device cpu")
            sys.exit(1)
        
        try:
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA is available. Using device: {device_name}")
        except Exception as e:
            print(f"Warning: CUDA is available but encountered an error getting device name: {e}")
            print("This might indicate an issue with your CUDA installation.")
    else:
        print("Using CPU for inference.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Phi-4-Mini-Reasoning benchmarks with different configurations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--prompt_sets", type=str, nargs="+", default=["all", "short", "medium", "long", "reasoning", "creative"],
                        help="Prompt sets to benchmark (default: all sets)")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per prompt for averaging")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results after benchmarking")
    return parser.parse_args()

def main():
    """Main function to run benchmarks with different configurations."""
    args = parse_args()
    
    # Check if CUDA is available when requested
    check_cuda_availability(args.device)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Generate timestamp for this benchmark run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Map of configurations to run
    configurations = []
    
    # Add all prompt sets or individual ones
    if "all" in args.prompt_sets:
        configurations.append({
            "name": "all_prompts",
            "prompt_types": ["short", "medium", "long", "reasoning", "creative"],
            "output": f"results/benchmark_all_{timestamp}.json"
        })
    else:
        for prompt_set in args.prompt_sets:
            if prompt_set != "all":
                configurations.append({
                    "name": prompt_set,
                    "prompt_types": [prompt_set],
                    "output": f"results/benchmark_{prompt_set}_{timestamp}.json"
                })
    
    # Run each configuration
    for config in configurations:
        print(f"\n{'='*80}")
        print(f"Running benchmark for {config['name']} prompts")
        print(f"{'='*80}")
        
        # Build command
        cmd = [
            "python", "benchmark.py",
            "--model", "microsoft/Phi-4-mini-reasoning",
            "--device", args.device,
            "--prompt_types", *config["prompt_types"],
            "--num_runs", str(args.num_runs),
            "--max_new_tokens", str(args.max_new_tokens),
            "--output", config["output"]
        ]
        
        # Run benchmark
        subprocess.run(cmd)
        
        # Visualize results if requested
        if args.visualize:
            print(f"\nVisualizing results for {config['name']} prompts")
            subprocess.run([
                "python", "visualize_results.py",
                "--results_file", config["output"]
            ])
    
    print("\nAll benchmarks complete!")

if __name__ == "__main__":
    main()
