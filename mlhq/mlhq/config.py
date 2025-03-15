import json
import os
from importlib import resources

def load_model_registry():
    """Dynamically load model-registry.json at runtime."""
    try:
        with resources.open_text("mlhq", "model-registry.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model registry: {e}")
        return {}

# No preloading – Call load_model_registry() dynamically where needed
