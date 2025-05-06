"""
Test script to verify the model caching system is working correctly.
This script will:
1. Check if a model is already cached
2. Download and cache a model if requested
3. Load a model from cache if available
"""

import os
import argparse
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def clean_model_name_for_path(model_name):
    """Clean model name to be used in file paths."""
    return model_name.replace('/', '_').replace('-', '_')

def get_model_cache_path(model_name):
    """Get the path where the model should be cached."""
    clean_name = clean_model_name_for_path(model_name)
    return os.path.join("models", clean_name)

def is_model_cached(model_name):
    """Check if the model is already cached locally."""
    cache_path = get_model_cache_path(model_name)
    return os.path.exists(cache_path) and len(os.listdir(cache_path)) > 0

def cache_model(model_name):
    """Download and cache the model locally."""
    cache_path = get_model_cache_path(model_name)
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_path, exist_ok=True)
    
    print(f"Downloading and caching model {model_name} to {cache_path}...")
    
    # Download model and tokenizer to the cache directory
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        local_files_only=False,
        cache_dir=cache_path
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        cache_dir=cache_path
    )
    
    print(f"Model and tokenizer cached successfully to {cache_path}")
    return model, tokenizer

def load_cached_model(model_name):
    """Load a model from the local cache."""
    cache_path = get_model_cache_path(model_name)
    
    if not is_model_cached(model_name):
        print(f"Error: Model {model_name} is not cached at {cache_path}")
        return None, None
    
    print(f"Loading model {model_name} from cache at {cache_path}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            local_files_only=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            cache_path,
            local_files_only=True
        )
        
        print(f"Model and tokenizer loaded successfully from {cache_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from cache: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Test model caching system")
    parser.add_argument("--model", type=str, default="microsoft/Phi-4-mini-reasoning", 
                        help="Model to test caching with")
    parser.add_argument("--cache", action="store_true", 
                        help="Download and cache the model")
    parser.add_argument("--load", action="store_true", 
                        help="Load the model from cache")
    parser.add_argument("--check", action="store_true", 
                        help="Check if the model is cached")
    parser.add_argument("--clear", action="store_true", 
                        help="Clear the model cache")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    if args.check:
        cached = is_model_cached(args.model)
        print(f"Model {args.model} is {'already cached' if cached else 'not cached'}")
        if cached:
            print(f"Cache location: {get_model_cache_path(args.model)}")
            print(f"Cache contents: {os.listdir(get_model_cache_path(args.model))}")
    
    if args.clear:
        cache_path = get_model_cache_path(args.model)
        if os.path.exists(cache_path):
            print(f"Clearing cache for model {args.model} at {cache_path}...")
            shutil.rmtree(cache_path)
            print("Cache cleared successfully")
        else:
            print(f"No cache found for model {args.model}")
    
    if args.cache:
        model, tokenizer = cache_model(args.model)
        if model is not None:
            print("Model cached successfully")
            # Test the model with a simple prompt
            prompt = "Hello, my name is"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20)
            print(f"Test generation: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    if args.load:
        model, tokenizer = load_cached_model(args.model)
        if model is not None:
            print("Model loaded successfully from cache")
            # Test the model with a simple prompt
            prompt = "Hello, my name is"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20)
            print(f"Test generation: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
