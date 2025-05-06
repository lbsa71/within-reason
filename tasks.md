# Phi-4-Mini-Reasoning Benchmarking Tasks

## Current Tasks
- [x] Set up virtual environment
- [x] Install required packages (transformers, torch, accelerate)
- [x] Create benchmarking script
- [x] Define a set of diverse prompts for testing
- [x] Create helper scripts for running benchmarks
- [x] Add CUDA availability checks
- [x] Improve CUDA error handling with helpful messages
- [x] Add model caching system
- [x] Add logging of generated answers
- [x] Add quantization support for large models
- [x] Improve output quality with proper chat formatting
- [ ] Run benchmarks and measure tokens/second
- [ ] Generate report with results

## Completed Tasks
- [x] Set up virtual environment (phi4_benchmark_env)
- [x] Create prompts.py with various prompt categories (short, medium, long, reasoning, creative)
- [x] Create benchmark.py for measuring tokens/second performance
- [x] Create visualize_results.py for analyzing and visualizing benchmark results
- [x] Create run_benchmarks.py helper script for different configurations
- [x] Create run_benchmark.bat for easy execution on Windows
- [x] Add CUDA availability checks to all scripts
- [x] Improve error handling for CUDA availability checks
- [x] Add automatic installation of CUDA-enabled PyTorch when needed
- [x] Implement model caching system with local model usage option
- [x] Add logging of generated answers in separate JSON files
- [x] Include model name in output filenames
- [x] Add 4-bit and 8-bit quantization support for running large models on limited VRAM
- [x] Add system prompts and chat formatting for better model responses
- [x] Implement response extraction to remove prompt repetition in outputs

## Notes
- Using Phi-4-Mini-Reasoning model from Microsoft
- Benchmarking will measure tokens per second performance
- Will test with various prompt lengths and complexities
- Results will be saved in JSON format and can be visualized with charts
- Supports both pipeline API and direct model usage
- Can run on either CPU or CUDA-enabled GPU
- Scripts will exit with error if CUDA is requested but not available
- Provides helpful instructions for installing CUDA-enabled PyTorch
- Models can be cached locally to avoid repeated downloads
- Generated answers are logged in separate JSON files for analysis
- Supports 4-bit and 8-bit quantization for running large models on GPUs with limited VRAM
- Uses system prompts and proper chat formatting for better quality responses
