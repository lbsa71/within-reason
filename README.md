# within-reason

## Phi-4-Mini-Reasoning Benchmarking

This project benchmarks the Microsoft Phi-4-Mini-Reasoning model to measure its performance in tokens per second across various prompts and scenarios.

### Project Structure
- `benchmark.py`: Main script for running benchmarks
- `prompts.py`: Collection of test prompts with varying complexity
- `visualize_results.py`: Script for creating visualizations of benchmark results
- `run_benchmark.bat`: Windows batch script for easy execution
- `results/`: Directory containing benchmark results and generated answers
- `models/`: Directory for cached model files

### Setup
```bash
# Create and activate virtual environment (Windows)
python -m venv phi4_benchmark_env
.\phi4_benchmark_env\Scripts\activate

# Install dependencies
pip install transformers torch accelerate matplotlib numpy
```

### Usage
The easiest way to run the benchmarks is using the batch file:

```bash
# Run with default settings (CUDA, all prompt types, 3 runs per prompt)
.\run_benchmark.bat

# Run on CPU only
.\run_benchmark.bat --cpu

# Run with specific prompt types
.\run_benchmark.bat --short --medium

# Run with a different model
.\run_benchmark.bat --model "microsoft/phi-3-mini-4k-instruct"

# Run with model caching options
.\run_benchmark.bat --use-local  # Use only locally cached model
.\run_benchmark.bat --no-cache   # Don't cache the model

# Run with visualization
.\run_benchmark.bat --visualize
```

Or run the Python script directly:

```bash
python benchmark.py --device cuda --prompt_types all --num_runs 3 --max_new_tokens 512 --model "microsoft/Phi-4-mini-reasoning" --cache_model
```

### Features
- Benchmarks tokens per second performance across different prompt types
- Supports both CPU and CUDA-enabled GPU execution
- Automatically checks for CUDA availability and provides helpful error messages
- Caches models locally to avoid repeated downloads
- Logs generated answers in separate JSON files for analysis
- Includes model name in output filenames for easy identification
- Creates visualizations of benchmark results

See `tasks.md` for current project status and roadmap.