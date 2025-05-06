"""
Benchmark script for measuring the performance of the Phi-4-Mini-Reasoning model.
Measures tokens per second across various prompts and scenarios.
"""

import os
import time
import json
import argparse
import sys
from datetime import datetime
from statistics import mean, median, stdev
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompts import ALL_PROMPTS

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

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
            print("python benchmark.py --device cpu")
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
    parser = argparse.ArgumentParser(description="Benchmark Phi-4-Mini-Reasoning model")
    parser.add_argument("--model", type=str, default="microsoft/Phi-4-mini-reasoning",
                        help="Model identifier (default: microsoft/Phi-4-mini-reasoning)")
    parser.add_argument("--prompt_types", type=str, nargs="+", 
                        default=["short", "medium", "long", "reasoning", "creative"],
                        help="Types of prompts to use for benchmarking")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per prompt for averaging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--output", type=str, default=f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        help="Output file for benchmark results")
    parser.add_argument("--use_pipeline", action="store_true",
                        help="Use pipeline API instead of direct model calls")
    return parser.parse_args()

def benchmark_pipeline(model_id, prompts, max_new_tokens, device, num_runs):
    """Benchmark using the pipeline API."""
    print(f"Loading model {model_id} using pipeline API on {device}...")
    pipe = pipeline("text-generation", model=model_id, device=device)
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Running prompt {i+1}/{len(prompts)}")
        prompt_results = []
        
        for run in range(num_runs):
            # Tokenize to get input token count
            input_text = prompt["content"]
            input_tokens = pipe.tokenizer(input_text, return_tensors="pt")
            input_token_count = input_tokens.input_ids.shape[1]
            
            # Time the generation
            start_time = time.time()
            output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            end_time = time.time()
            
            # Calculate output token count (approximately)
            output_text = output[0]["generated_text"]
            output_tokens = pipe.tokenizer(output_text, return_tensors="pt")
            output_token_count = output_tokens.input_ids.shape[1]
            generated_tokens = output_token_count - input_token_count
            
            # Calculate tokens per second
            elapsed_time = end_time - start_time
            tokens_per_second = generated_tokens / elapsed_time
            
            prompt_results.append({
                "run": run + 1,
                "input_tokens": input_token_count,
                "generated_tokens": generated_tokens,
                "total_tokens": output_token_count,
                "time_seconds": elapsed_time,
                "tokens_per_second": tokens_per_second
            })
            
            print(f"  Run {run+1}: {tokens_per_second:.2f} tokens/sec ({generated_tokens} tokens in {elapsed_time:.2f}s)")
        
        # Calculate average performance for this prompt
        avg_tokens_per_second = mean([r["tokens_per_second"] for r in prompt_results])
        results.append({
            "prompt": prompt["content"],
            "runs": prompt_results,
            "avg_tokens_per_second": avg_tokens_per_second
        })
        
        print(f"  Average: {avg_tokens_per_second:.2f} tokens/sec\n")
    
    return results

def benchmark_direct(model_id, prompts, max_new_tokens, device, num_runs):
    """Benchmark using direct model and tokenizer calls."""
    print(f"Loading model {model_id} directly on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Running prompt {i+1}/{len(prompts)}")
        prompt_results = []
        
        for run in range(num_runs):
            # Format the prompt for the model
            input_text = prompt["content"]
            
            # Tokenize
            input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
            input_token_count = input_tokens.input_ids.shape[1]
            
            # Time the generation
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    input_tokens.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            end_time = time.time()
            
            # Calculate tokens generated
            generated_tokens = output.shape[1] - input_token_count
            
            # Calculate tokens per second
            elapsed_time = end_time - start_time
            tokens_per_second = generated_tokens / elapsed_time
            
            # Decode output for reference
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            prompt_results.append({
                "run": run + 1,
                "input_tokens": input_token_count,
                "generated_tokens": generated_tokens,
                "total_tokens": output.shape[1],
                "time_seconds": elapsed_time,
                "tokens_per_second": tokens_per_second
            })
            
            print(f"  Run {run+1}: {tokens_per_second:.2f} tokens/sec ({generated_tokens} tokens in {elapsed_time:.2f}s)")
        
        # Calculate average performance for this prompt
        avg_tokens_per_second = mean([r["tokens_per_second"] for r in prompt_results])
        results.append({
            "prompt": prompt["content"],
            "runs": prompt_results,
            "avg_tokens_per_second": avg_tokens_per_second
        })
        
        print(f"  Average: {avg_tokens_per_second:.2f} tokens/sec\n")
    
    return results

def summarize_results(all_results, model_id, device):
    """Generate a summary of the benchmark results."""
    all_tps = [prompt_result["avg_tokens_per_second"] for prompt_result in all_results]
    
    summary = {
        "model": model_id,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "overall_stats": {
            "mean_tokens_per_second": mean(all_tps),
            "median_tokens_per_second": median(all_tps),
            "min_tokens_per_second": min(all_tps),
            "max_tokens_per_second": max(all_tps),
            "stddev_tokens_per_second": stdev(all_tps) if len(all_tps) > 1 else 0,
        },
        "detailed_results": all_results
    }
    
    return summary

def main():
    """Main function to run benchmarks."""
    args = parse_args()
    
    # Check if CUDA is available when requested
    check_cuda_availability(args.device)
    
    # Collect prompts to benchmark
    prompts_to_benchmark = []
    for prompt_type in args.prompt_types:
        if prompt_type in ALL_PROMPTS:
            prompts_to_benchmark.extend(ALL_PROMPTS[prompt_type])
        else:
            print(f"Warning: Unknown prompt type '{prompt_type}', skipping")
    
    print(f"Benchmarking {args.model} on {args.device}")
    print(f"Running {len(prompts_to_benchmark)} prompts, {args.num_runs} times each")
    
    # Run benchmarks
    if args.use_pipeline:
        results = benchmark_pipeline(args.model, prompts_to_benchmark, args.max_new_tokens, args.device, args.num_runs)
    else:
        results = benchmark_direct(args.model, prompts_to_benchmark, args.max_new_tokens, args.device, args.num_runs)
    
    # Summarize and save results
    summary = summarize_results(results, args.model, args.device)
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {args.output}")
    print(f"Overall average: {summary['overall_stats']['mean_tokens_per_second']:.2f} tokens/sec")
    print(f"Min: {summary['overall_stats']['min_tokens_per_second']:.2f}, Max: {summary['overall_stats']['max_tokens_per_second']:.2f}")

if __name__ == "__main__":
    main()
