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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from prompts import ALL_PROMPTS
import re

# Print system information for debugging
print("Python version:", platform.python_version())
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    try:
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
    except Exception as e:
        print("Error getting CUDA device info:", e)
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
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per prompt")
    parser.add_argument("--use_pipeline", action="store_true",
                        help="Use the pipeline API instead of direct model calls")
    parser.add_argument("--output", type=str, default="",
                        help="Output file for results (default: auto-generated)")
    parser.add_argument("--use_local_model", action="store_true",
                        help="Use locally cached model files only (no downloads)")
    parser.add_argument("--cache_model", action="store_true",
                        help="Cache the model locally for future runs")
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=0,
                        help="Quantize model to specified bit width (0=no quantization, 4=4-bit, 8=8-bit)")
    return parser.parse_args()

def extract_clean_response(full_output, model_id):
    """Extract just the model's final answer, removing system prompts, user prompts, and thinking."""
    # First try to extract any content within <think> tags
    thinking = ""
    clean_response = full_output
    
    # Extract thinking if present
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove the thinking part from the response
        clean_response = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    
    # Try to remove system and user prompts
    # Common patterns to identify the start of the actual response
    patterns = [
        r'system\s*\n.*?\n\s*user\s*\n.*?\n\s*assistant\s*\n(.*)',  # system\n...\nuser\n...\nassistant\n...
        r'<\|im_start\|>assistant\s*\n(.*?)(?:<\|im_end\|>|$)',     #  assistant\n...
        r'assistant[:\s]*\n(.*)',                                   # assistant:\n... or assistant\n...
        r'assistant[:\s]*(.*)',                                     # assistant: ... or assistant ...
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_response, re.DOTALL)
        if match:
            clean_response = match.group(1).strip()
            break
    
    # If we still have the full output, try to extract just the last part
    if clean_response == full_output:
        # Try to get just the last paragraph
        parts = clean_response.split('\n\n')
        if len(parts) > 1:
            clean_response = parts[-1].strip()
    
    # Create a structured response
    result = {
        "response": clean_response
    }
    
    # Add thinking if present
    if thinking:
        result["thinking"] = thinking
    
    # Add full output for debugging
    result["full_output"] = full_output
    
    return result

def benchmark_pipeline(model_id, prompts, max_new_tokens, device, num_runs, use_local_model=False, cache_model=False, quantize=0):
    """Benchmark using the pipeline API."""
    results = []
    generated_answers = []
    
    # Check if model is already cached and we want to use local model only
    if use_local_model and not is_model_cached(model_id):
        print("Error: Model %s is not cached locally and --use_local_model is specified." % model_id)
        print("Please run without --use_local_model first to download the model.")
        sys.exit(1)
    
    # Set up cache directory
    cache_dir = None
    if cache_model or use_local_model:
        cache_dir = get_model_cache_dir(model_id)
        os.makedirs(cache_dir, exist_ok=True)
        print("Using cache directory:", cache_dir)
    
    try:
        # Set up quantization parameters
        model_kwargs = {}
        quantization_config = None
        if quantize > 0:
            try:
                # Import bitsandbytes for quantization
                import bitsandbytes as bnb
                if quantize == 8:
                    print("Loading model with 8-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                elif quantize == 4:
                    print("Loading model with 4-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("Warning: bitsandbytes not installed. Quantization disabled.")
                print("Install with: pip install bitsandbytes")
                quantize = 0
        
        # Create the pipeline
        print("Loading model and tokenizer...")
        start_time = time.time()
        
        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
            model_kwargs=model_kwargs,
            cache_dir=cache_dir,
            local_files_only=use_local_model
        )
        
        load_time = time.time() - start_time
        print("Model loaded in %.2f seconds" % load_time)
        
        # System prompt for better responses
        system_prompt = "You are a helpful, accurate, and concise assistant. Answer the user's questions directly without repeating the question. Provide factual information and avoid unnecessary text."
        
        # Run benchmarks
        for i, prompt in enumerate(prompts):
            # Extract content from prompt if it's a dictionary
            prompt_text = prompt["content"] if isinstance(prompt, dict) and "content" in prompt else prompt
            
            print("Running prompt %d/%d: %s" % (i+1, len(prompts), prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text))
            prompt_results = []
            best_output = None
            best_tps = 0
            
            for run in range(num_runs):
                print("  Run %d/%d..." % (run+1, num_runs), end="", flush=True)
                
                # Format prompt with chat template if possible
                try:
                    formatted_prompt = pipe.tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, 
                         {"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # Fallback if chat template not supported
                    formatted_prompt = f"{system_prompt}\n\nUser: {prompt_text}\n\nAssistant:"
                
                # Tokenize the prompt to get input length
                input_length = len(pipe.tokenizer.encode(formatted_prompt))
                
                # Run the model
                start_time = time.time()
                output = pipe(formatted_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
                end_time = time.time()
                
                # Get the generated text
                full_output = output[0]["generated_text"]
                
                # Extract just the assistant's response (remove the prompt part)
                extracted_output = extract_clean_response(full_output, model_id)
                
                # Calculate output length and tokens per second
                output_length = len(pipe.tokenizer.encode(full_output))
                new_tokens = output_length - input_length
                elapsed_time = end_time - start_time
                tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0
                
                print(" %.2f tokens/sec (%d tokens in %.2f seconds)" % (tokens_per_second, new_tokens, elapsed_time))
                
                # Store result
                result = {
                    "prompt": prompt_text,
                    "elapsed_time": elapsed_time,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "new_tokens": new_tokens,
                    "tokens_per_second": tokens_per_second
                }
                prompt_results.append(result)
                
                # Track best output
                if tokens_per_second > best_tps:
                    best_tps = tokens_per_second
                    best_output = extracted_output
            
            # Store the best output for this prompt
            generated_answers.append({
                "prompt": prompt_text,
                "result": best_output
            })
            
            # Add all runs for this prompt to results
            results.append({
                "prompt": prompt_text,
                "runs": prompt_results
            })
    
    except Exception as e:
        print("Error during benchmark:", e)
        import traceback
        traceback.print_exc()
    
    return results, generated_answers

def benchmark_direct(model_id, prompts, max_new_tokens, device, num_runs, use_local_model=False, cache_model=False, quantize=0):
    """Benchmark using direct model and tokenizer calls."""
    results = []
    generated_answers = []
    
    # Check if model is already cached and we want to use local model only
    if use_local_model and not is_model_cached(model_id):
        print("Error: Model %s is not cached locally and --use_local_model is specified." % model_id)
        print("Please run without --use_local_model first to download the model.")
        sys.exit(1)
    
    # Set up cache directory
    cache_dir = None
    if cache_model or use_local_model:
        cache_dir = get_model_cache_dir(model_id)
        os.makedirs(cache_dir, exist_ok=True)
        print("Using cache directory:", cache_dir)
    
    try:
        # Set up quantization parameters
        model_kwargs = {}
        quantization_config = None
        if quantize > 0:
            try:
                # Import bitsandbytes for quantization
                import bitsandbytes as bnb
                if quantize == 8:
                    print("Loading model with 8-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                elif quantize == 4:
                    print("Loading model with 4-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("Warning: bitsandbytes not installed. Quantization disabled.")
                print("Install with: pip install bitsandbytes")
                quantize = 0
        
        # Load the model and tokenizer
        print("Loading tokenizer...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir,
            local_files_only=use_local_model
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            cache_dir=cache_dir,
            local_files_only=use_local_model,
            **model_kwargs
        )
        
        load_time = time.time() - start_time
        print("Model and tokenizer loaded in %.2f seconds" % load_time)
        
        # System prompt for better responses
        system_prompt = "You are a helpful, accurate, and concise assistant. Answer the user's questions directly without repeating the question. Provide factual information and avoid unnecessary text."
        
        # Run benchmarks
        for i, prompt in enumerate(prompts):
            # Extract content from prompt if it's a dictionary
            prompt_text = prompt["content"] if isinstance(prompt, dict) and "content" in prompt else prompt
            
            print("Running prompt %d/%d: %s" % (i+1, len(prompts), prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text))
            prompt_results = []
            best_output = None
            best_tps = 0
            
            for run in range(num_runs):
                print("  Run %d/%d..." % (run+1, num_runs), end="", flush=True)
                
                # Format prompt with chat template if possible
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, 
                         {"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # Fallback if chat template not supported
                    formatted_prompt = f"{system_prompt}\n\nUser: {prompt_text}\n\nAssistant:"
                
                # Tokenize the prompt
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                input_length = inputs.input_ids.shape[1]
                
                # Run the model
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7
                    )
                end_time = time.time()
                
                # Get the generated text
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the assistant's response (remove the prompt part)
                extracted_output = extract_clean_response(full_output, model_id)
                
                # Calculate output length and tokens per second
                output_length = outputs.shape[1]
                new_tokens = output_length - input_length
                elapsed_time = end_time - start_time
                tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0
                
                print(" %.2f tokens/sec (%d tokens in %.2f seconds)" % (tokens_per_second, new_tokens, elapsed_time))
                
                # Store result
                result = {
                    "prompt": prompt_text,
                    "elapsed_time": elapsed_time,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "new_tokens": new_tokens,
                    "tokens_per_second": tokens_per_second
                }
                prompt_results.append(result)
                
                # Track best output
                if tokens_per_second > best_tps:
                    best_tps = tokens_per_second
                    best_output = extracted_output
            
            # Store the best output for this prompt
            generated_answers.append({
                "prompt": prompt_text,
                "result": best_output
            })
            
            # Add all runs for this prompt to results
            results.append({
                "prompt": prompt_text,
                "runs": prompt_results
            })
    
    except Exception as e:
        print("Error during benchmark:", e)
        import traceback
        traceback.print_exc()
    
    return results, generated_answers

def summarize_results(all_results, model_id, device):
    """Generate a summary of the benchmark results."""
    if not all_results:
        return None
    
    all_tps = []
    for result in all_results:
        for run in result["runs"]:
            all_tps.append(run["tokens_per_second"])
    
    if not all_tps:
        return None
    
    # Create summary
    summary = {
        "model": model_id,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
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
        if prompt_type == "all":
            for pt in ALL_PROMPTS:
                prompts_to_benchmark.extend(ALL_PROMPTS[pt])
            break
        elif prompt_type in ALL_PROMPTS:
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
    
    if args.quantize > 0:
        print(f"Using {args.quantize}-bit quantization")
        
        # Check if bitsandbytes is installed
        try:
            import bitsandbytes as bnb
            print("bitsandbytes version:", bnb.__version__)
        except ImportError:
            print("Warning: bitsandbytes not installed. Please install it with:")
            print("pip install bitsandbytes")
            print("Continuing without quantization...")
            args.quantize = 0
    
    # Run benchmarks
    if args.use_pipeline:
        results, generated_answers = benchmark_pipeline(
            args.model, 
            prompts_to_benchmark, 
            args.max_new_tokens, 
            actual_device, 
            args.num_runs,
            args.use_local_model,
            args.cache_model,
            args.quantize
        )
    else:
        results, generated_answers = benchmark_direct(
            args.model, 
            prompts_to_benchmark, 
            args.max_new_tokens, 
            actual_device, 
            args.num_runs,
            args.use_local_model,
            args.cache_model,
            args.quantize
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
