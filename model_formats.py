"""
Model-specific prompt formats for different LLM families.
This helps ensure proper formatting when the chat template isn't available.
"""

def get_model_specific_format(model_id):
    """Get a model-specific prompt format based on the model ID."""
    # Check for specific model families
    model_id_lower = model_id.lower()
    
    if "qwen" in model_id_lower:
        # Qwen-style format with special tokens
        return {
            "format": "\u200bassistant\n{system}\n\nassistant\n{prompt}\n\nassistant\n",
            "response_prefix": "\u200bassistant\n",
            "extract_pattern": "\u200bassistant\n(.*?)(?:\n|$)"
        }
    elif "phi" in model_id_lower:
        # Phi-style format
        return {
            "format": "<|system|>\n{system}\n\nassistant\n{prompt}\nassistant\n",
            "response_prefix": "assistant\n",
            "extract_pattern": "assistant\n(.*?)(?:\n|$)"
        }
