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
- `model_formats.py`: Model-specific prompt formatting information

### Setup
```bash
# Create and activate virtual environment (Windows)
python -m venv phi4_benchmark_env
.\phi4_benchmark_env\Scripts\activate

# Install dependencies
pip install transformers torch accelerate matplotlib numpy

# For quantization support (optional)
pip install bitsandbytes
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
.\run_benchmark.bat --model "meta-llama/Llama-3.1-8B"

> **Note:** For gated Hugging Face models (like Meta-Llama-3.1-8B), you must:
> 1. Request and receive access on the model's Hugging Face page.
> 2. Set your `HF_TOKEN` environment variable (or run `huggingface-cli login`).
> 3. Ensure your token has "Read" access. If you still get 401 errors, check your access approval, token, and login status.

# Run with model caching options
.\run_benchmark.bat --use-local  # Use only locally cached model
.\run_benchmark.bat --no-cache   # Don't cache the model

# Run with quantization for large models on limited VRAM
.\run_benchmark.bat --4bit       # Use 4-bit quantization
.\run_benchmark.bat --8bit       # Use 8-bit quantization

# Run with GPU optimizations for better utilization
.\run_benchmark.bat --optimize-gpu
.\run_benchmark.bat --batch 4    # Use batch size of 4

# Run with visualization
.\run_benchmark.bat --visualize
```

Or run the Python script directly:

```bash
python benchmark.py --device cuda --prompt_types all --num_runs 3 --max_new_tokens 512 --model "microsoft/Phi-4-mini-reasoning" --cache_model --quantize 4 --optimize_gpu --batch_size 2
```

### Features
- Benchmarks tokens per second performance across different prompt types
- Supports both CPU and CUDA-enabled GPU execution
- Automatically checks for CUDA availability and provides helpful error messages
- Caches models locally to avoid repeated downloads
- Logs generated answers in separate JSON files for analysis
- Includes model name in output filenames for easy identification
- Creates visualizations of benchmark results
- Supports 4-bit and 8-bit quantization for running large models on GPUs with limited VRAM
- GPU optimization options for better hardware utilization
- Structured output with separate response and thinking components
- Proper chat formatting with system prompts
- Llama-family models (TinyLlama, Llama, Vicuna) now use the canonical chat template:
  
  ```
  <|system|>
  {system}</s>
  <|user|>
  {prompt}</s>
  <|assistant|>
  ```
  This prevents prompt echoes and ensures high-quality, direct responses.

### Running Large Models on Limited VRAM
For larger models like Qwen3-14B (14.8B parameters), you'll need to use quantization:

- **16GB GPU**: Use 4-bit quantization with `--4bit` option
- **8GB GPU**: 4-bit quantization might work with additional optimizations
- **<8GB GPU**: Consider using CPU mode with `--cpu` (will be very slow)

The quantization feature reduces the precision of model weights, allowing larger models to fit in limited VRAM with minimal impact on output quality.

### Improving GPU Utilization
LLM inference often shows low GPU utilization due to the autoregressive nature of token generation. To improve utilization:

1. Use the `--optimize-gpu` flag to enable mixed precision and other optimizations
2. Try different batch sizes with `--batch <size>` (if running multiple prompts)
3. For best performance, consider using specialized inference engines like vLLM or TensorRT

See `tasks.md` for current project status and roadmap.