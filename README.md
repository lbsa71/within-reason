# within-reason

## Phi-4-Mini-Reasoning Benchmarking

This project benchmarks the Microsoft Phi-4-Mini-Reasoning model to measure its performance in tokens per second across various prompts and scenarios.

### Project Structure
- `benchmark.py`: Main script for running benchmarks
- `prompts.py`: Collection of test prompts with varying complexity
- `results/`: Directory containing benchmark results

### Setup
```bash
# Activate virtual environment
.\phi4_benchmark_env\Scripts\activate

# Install dependencies
pip install transformers torch accelerate
```

### Usage
```bash
python benchmark.py
```

See `tasks.md` for current project status and roadmap.