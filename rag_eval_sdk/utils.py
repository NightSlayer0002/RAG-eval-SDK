import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
