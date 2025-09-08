# utils.py
import json
import os

def save_class_indices(mapping, path="models/class_indices.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f)

def load_class_indices(path="models/class_indices.json"):
    with open(path, "r") as f:
        mapping = json.load(f)
    # invert to map index -> class name
    inv = {int(v): k for k, v in mapping.items()}
    return mapping, inv
