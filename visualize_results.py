"""
Utility script to visualize benchmark results from the Phi-4-Mini-Reasoning model.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Phi-4-Mini-Reasoning benchmark results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the JSON results file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save visualization outputs")
    return parser.parse_args()

def load_results(results_file):
    """Load benchmark results from a JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_visualizations(results, output_dir):
    """Create visualizations from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract data
    prompts = [r["prompt"][:50] + "..." if len(r["prompt"]) > 50 else r["prompt"] for r in results["detailed_results"]]
    avg_tps = [r["avg_tokens_per_second"] for r in results["detailed_results"]]
    
    # Sort by tokens per second for better visualization
    sorted_indices = np.argsort(avg_tps)
    prompts = [prompts[i] for i in sorted_indices]
    avg_tps = [avg_tps[i] for i in sorted_indices]
    
    # 1. Bar chart of tokens per second for each prompt
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(prompts)), avg_tps, align='center')
    plt.yticks(range(len(prompts)), prompts)
    plt.xlabel('Tokens per Second')
    plt.title(f'Phi-4-Mini-Reasoning Performance by Prompt\nModel: {results["model"]}, Device: {results["device"]}')
    
    # Add values to the end of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{avg_tps[i]:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prompt_performance_{timestamp}.png'))
    
    # 2. Histogram of tokens per second distribution
    plt.figure(figsize=(10, 6))
    plt.hist(avg_tps, bins=10, alpha=0.7, color='blue')
    plt.axvline(results["overall_stats"]["mean_tokens_per_second"], color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {results["overall_stats"]["mean_tokens_per_second"]:.2f}')
    plt.axvline(results["overall_stats"]["median_tokens_per_second"], color='green', linestyle='dashed', linewidth=2, 
                label=f'Median: {results["overall_stats"]["median_tokens_per_second"]:.2f}')
    plt.xlabel('Tokens per Second')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Tokens per Second\nModel: {results["model"]}, Device: {results["device"]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tps_distribution_{timestamp}.png'))
    
    # 3. Create a summary text file
    summary_file = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Benchmark Summary for {results['model']}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Date: {results['timestamp']}\n")
        f.write(f"Device: {results['device']}\n")
        if results['device'] == 'cuda':
            f.write(f"GPU: {results['system_info']['device_name']}\n")
        f.write(f"CUDA Version: {results['system_info']['cuda_version']}\n\n")
        
        f.write("Performance Statistics:\n")
        f.write(f"  Mean: {results['overall_stats']['mean_tokens_per_second']:.2f} tokens/sec\n")
        f.write(f"  Median: {results['overall_stats']['median_tokens_per_second']:.2f} tokens/sec\n")
        f.write(f"  Min: {results['overall_stats']['min_tokens_per_second']:.2f} tokens/sec\n")
        f.write(f"  Max: {results['overall_stats']['max_tokens_per_second']:.2f} tokens/sec\n")
        f.write(f"  Standard Deviation: {results['overall_stats']['stddev_tokens_per_second']:.2f} tokens/sec\n\n")
        
        f.write("Top 5 Fastest Prompts:\n")
        for i in range(-1, -6, -1):
            if abs(i) <= len(avg_tps):
                f.write(f"  {avg_tps[i]:.2f} tokens/sec: {prompts[i]}\n")
        
        f.write("\nBottom 5 Slowest Prompts:\n")
        for i in range(5):
            if i < len(avg_tps):
                f.write(f"  {avg_tps[i]:.2f} tokens/sec: {prompts[i]}\n")
    
    print(f"Visualizations and summary saved to {output_dir}")
    print(f"Summary file: {summary_file}")

def main():
    """Main function to visualize benchmark results."""
    args = parse_args()
    results = load_results(args.results_file)
    create_visualizations(results, args.output_dir)

if __name__ == "__main__":
    main()
