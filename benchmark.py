"""
Benchmark script for measuring the performance of the Phi-4-Mini-Reasoning model.
Measures tokens per second across various prompts and scenarios.
"""

import os
import time
import json
import argparse
import sys
import platform
import shutil
from datetime import datetime
from statistics import mean, median, stdev
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompts import ALL_PROMPTS

# Print system information for debugging
print("Python version:", platform.python_version())
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device:", torch.cuda.get_device_name(0))
print("Running on:", platform.system(), platform.release())
print("")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def check_cuda_availability(device):
    """Check if CUDA is available when requested."""
    if device == "cuda":
        # Check if torch.cuda.is_available() returns True
        cuda_available = torch.cuda.is_available()
        print("CUDA available:", cuda_available)
        
        if not cuda_available:
            print("ERROR: CUDA device requested but CUDA is not available.")
            print("Your PyTorch installation does not have CUDA support.")
            print("\nTo fix this, reinstall PyTorch with CUDA support:")
            print("pip uninstall -y torch")
            print("pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("\nOr run the benchmark on CPU instead:")
            print("python benchmark.py --device cpu")
            sys.exit(1)
        
        # Check if we can actually use CUDA by creating a small tensor
        try:
            # Try to create a CUDA tensor to verify CUDA is working
            test_tensor = torch.tensor([1.0], device="cuda")
            del test_tensor  # Clean up
            
            # Get device name
            device_name = torch.cuda.get_device_name(0)
            print("CUDA is available and working. Using device:", device_name)
        except Exception as e:
            print("WARNING: CUDA is reported as available but encountered an error:", e)
            print("This might indicate an issue with your CUDA installation.")
            print("Falling back to CPU.")
            return "cpu"
    else:
        print("Using CPU for inference.")
    
    return device

def get_model_name_for_filename(model_id):
    """Extract a clean model name for use in filenames."""
    # Remove any path separators and replace with underscores
    clean_name = model_id.replace('/', '_').replace('\\', '_')
    # Remove any special characters that might cause issues in filenames
    clean_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in clean_name)
    return clean_name

def get_model_cache_dir(model_id):
    """Get the local cache directory for a model."""
    model_name = get_model_name_for_filename(model_id)
    return os.path.join("models", model_name)

def is_model_cached(model_id):
    """Check if the model is already cached locally."""
    cache_dir = get_model_cache_dir(model_id)
    # Check if the directory exists and has files in it
    return os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0

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
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for benchmark results (default: auto-generated)")
    parser.add_argument("--use_pipeline", action="store_true",
                        help="Use pipeline API instead of direct model calls")
    parser.add_argument("--use_local_model", action="store_true",
                        help="Use locally cached model files only (no downloads)")
    parser.add_argument("--cache_model", action="store_true",
                        help="Cache the model locally for future runs")
    return parser.parse_args()

def benchmark_pipeline(model_id, prompts, max_new_tokens, device, num_runs, use_local_model=False, cache_model=False):
    """Benchmark using the pipeline API."""
    cache_dir = get_model_cache_dir(model_id)
    local_files_only = use_local_model and is_model_cached(model_id)
    
    if use_local_model and not is_model_cached(model_id):
        print("WARNING: Local model requested but model is not cached.")
        print("Will download the model and cache it for future runs.")
        local_files_only = False
    
    print("Loading model %s using pipeline API on %s..." % (model_id, device))
    print("Using cache directory:", cache_dir)
    print("Local files only:", local_files_only)
    
    try:
        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            device=device,
            cache_dir=cache_dir if cache_model else None,
            local_files_only=local_files_only
        )
        
        # If caching was requested, save the model files
        if cache_model and not is_model_cached(model_id):
            print("Model loaded successfully. Caching for future runs...")
            # The model is already cached by HuggingFace in the cache_dir
    except Exception as e:
        print("Error loading model:", e)
        return [], []
    
    results = []
    generated_answers = []
    
    for i, prompt in enumerate(prompts):
        print("Running prompt %d/%d" % (i+1, len(prompts)))
        prompt_results = []
        best_output = None
        
        for run in range(num_runs):
            try:
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
                
                # Save the result
                run_result = {
                    "run": run + 1,
                    "input_tokens": input_token_count,
                    "generated_tokens": generated_tokens,
                    "total_tokens": output_token_count,
                    "time_seconds": elapsed_time,
                    "tokens_per_second": tokens_per_second
                }
                prompt_results.append(run_result)
                
                # Keep track of the best output (fastest generation)
                if best_output is None or tokens_per_second > best_output["tokens_per_second"]:
                    best_output = {
                        "tokens_per_second": tokens_per_second,
                        "output_text": output_text
                    }
                
                print("  Run %d: %.2f tokens/sec (%d tokens in %.2f s)" % (run+1, tokens_per_second, generated_tokens, elapsed_time))
            except Exception as e:
                print("  Error in run %d:" % (run+1), e)
                continue
        
        if not prompt_results:
            print("  No successful runs for prompt %d" % (i+1))
            continue
            
        # Calculate average performance for this prompt
        avg_tokens_per_second = mean([r["tokens_per_second"] for r in prompt_results])
        results.append({
            "prompt": prompt["content"],
            "runs": prompt_results,
            "avg_tokens_per_second": avg_tokens_per_second
        })
        
        # Save the generated answer
        generated_answers.append({
            "prompt": prompt["content"],
            "result": best_output["output_text"] if best_output else "No successful generation"
        })
        
        print("  Average: %.2f tokens/sec\n" % avg_tokens_per_second)
    
    return results, generated_answers

def benchmark_direct(model_id, prompts, max_new_tokens, device, num_runs, use_local_model=False, cache_model=False):
    """Benchmark using direct model and tokenizer calls."""
    cache_dir = get_model_cache_dir(model_id)
    local_files_only = use_local_model and is_model_cached(model_id)
    
    if use_local_model and not is_model_cached(model_id):
        print("WARNING: Local model requested but model is not cached.")
        print("Will download the model and cache it for future runs.")
        local_files_only = False
    
    print("Loading model %s directly on %s..." % (model_id, device))
    print("Using cache directory:", cache_dir)
    print("Local files only:", local_files_only)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir if cache_model else None,
            local_files_only=local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=cache_dir if cache_model else None,
            local_files_only=local_files_only
        )
        model.to(device)
        
        # If caching was requested, save the model files
        if cache_model and not is_model_cached(model_id):
            print("Model loaded successfully. Caching for future runs...")
            # The model is already cached by HuggingFace in the cache_dir
    except Exception as e:
        print("Error loading model:", e)
        return [], []
    
    results = []
    generated_answers = []
    
    for i, prompt in enumerate(prompts):
        print("Running prompt %d/%d" % (i+1, len(prompts)))
        prompt_results = []
        best_output = None
        
        for run in range(num_runs):
            try:
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
                
                # Save the result
                run_result = {
                    "run": run + 1,
                    "input_tokens": input_token_count,
                    "generated_tokens": generated_tokens,
                    "total_tokens": output.shape[1],
                    "time_seconds": elapsed_time,
                    "tokens_per_second": tokens_per_second
                }
                prompt_results.append(run_result)
                
                # Keep track of the best output (fastest generation)
                if best_output is None or tokens_per_second > best_output["tokens_per_second"]:
                    best_output = {
                        "tokens_per_second": tokens_per_second,
                        "output_text": output_text
                    }
                
                print("  Run %d: %.2f tokens/sec (%d tokens in %.2f s)" % (run+1, tokens_per_second, generated_tokens, elapsed_time))
            except Exception as e:
                print("  Error in run %d:" % (run+1), e)
                continue
        
        if not prompt_results:
            print("  No successful runs for prompt %d" % (i+1))
            continue
            
        # Calculate average performance for this prompt
        avg_tokens_per_second = mean([r["tokens_per_second"] for r in prompt_results])
        results.append({
            "prompt": prompt["content"],
            "runs": prompt_results,
            "avg_tokens_per_second": avg_tokens_per_second
        })
        
        # Save the generated answer
        generated_answers.append({
            "prompt": prompt["content"],
            "result": best_output["output_text"] if best_output else "No successful generation"
        })
        
        print("  Average: %.2f tokens/sec\n" % avg_tokens_per_second)
    
    return results, generated_answers

def summarize_results(all_results, model_id, device):
    """Generate a summary of the benchmark results."""
    if not all_results:
        print("No benchmark results to summarize.")
        return None
        
    all_tps = [prompt_result["avg_tokens_per_second"] for prompt_result in all_results]
    
    summary = {
        "model": model_id,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "os": "%s %s" % (platform.system(), platform.release())
        },
        "overall_stats": {
            "mean_tokens_per_second": mean(all_tps) if all_tps else 0,
            "median_tokens_per_second": median(all_tps) if all_tps else 0,
            "min_tokens_per_second": min(all_tps) if all_tps else 0,
            "max_tokens_per_second": max(all_tps) if all_tps else 0,
            "stddev_tokens_per_second": stdev(all_tps) if len(all_tps) > 1 else 0,
        },
        "detailed_results": all_results
    }
    
    return summary

def main():
    """Main function to run benchmarks."""
    args = parse_args()
    
    # Check if CUDA is available when requested and get the actual device to use
    actual_device = check_cuda_availability(args.device)
    
    # Get a clean model name for the filename
    model_name = get_model_name_for_filename(args.model)
    
    # Generate output filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_types_str = '_'.join(args.prompt_types)
    
    if args.output:
        output_file = args.output
    else:
        output_file = "results/benchmark_%s_%s_%s.json" % (model_name, prompt_types_str, timestamp)
    
    # Generate answers output filename
    answers_file = "results/answers_%s_%s_%s.json" % (model_name, prompt_types_str, timestamp)
    
    # Collect prompts to benchmark
    prompts_to_benchmark = []
    for prompt_type in args.prompt_types:
        if prompt_type in ALL_PROMPTS:
            prompts_to_benchmark.extend(ALL_PROMPTS[prompt_type])
        else:
            print("Warning: Unknown prompt type '%s', skipping" % prompt_type)
    
    if not prompts_to_benchmark:
        print("Error: No valid prompts to benchmark.")
        sys.exit(1)
    
    print("Benchmarking %s on %s" % (args.model, actual_device))
    print("Running %d prompts, %d times each" % (len(prompts_to_benchmark), args.num_runs))
    print("Results will be saved to: %s" % output_file)
    print("Generated answers will be saved to: %s" % answers_file)
    
    if args.use_local_model:
        print("Using locally cached model (if available)")
    
    if args.cache_model:
        print("Caching model for future runs in:", get_model_cache_dir(args.model))
    
    # Run benchmarks
    if args.use_pipeline:
        results, generated_answers = benchmark_pipeline(
            args.model, 
            prompts_to_benchmark, 
            args.max_new_tokens, 
            actual_device, 
            args.num_runs,
            args.use_local_model,
            args.cache_model
        )
    else:
        results, generated_answers = benchmark_direct(
            args.model, 
            prompts_to_benchmark, 
            args.max_new_tokens, 
            actual_device, 
            args.num_runs,
            args.use_local_model,
            args.cache_model
        )
    
    # Check if we got any results
    if not results:
        print("Error: No benchmark results were generated.")
        sys.exit(1)
    
    # Summarize and save results
    summary = summarize_results(results, args.model, actual_device)
    if not summary:
        print("Error: Could not generate summary from results.")
        sys.exit(1)
    
    # Save results to file
    try:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print("\nBenchmark complete! Results saved to %s" % output_file)
        
        # Save generated answers to a separate file
        with open(answers_file, 'w') as f:
            json.dump(generated_answers, f, indent=2)
        print("Generated answers saved to %s" % answers_file)
    except Exception as e:
        print("Error saving results:", e)
        sys.exit(1)
    
    print("Overall average: %.2f tokens/sec" % summary['overall_stats']['mean_tokens_per_second'])
    print("Min: %.2f, Max: %.2f" % (summary['overall_stats']['min_tokens_per_second'], summary['overall_stats']['max_tokens_per_second']))

if __name__ == "__main__":
    main()
