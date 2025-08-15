import json
from pathlib import Path

def load_json(default_path: str, config_input: str) -> dict:
    path = Path(config_input)

    if not path.is_file():
        path = Path(default_path) / f"{config_input}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_input}.json in {default_path}")

    try:
        with open(path, "r") as f:
            return json.load(f)

    except Exception as e:
        raise ValueError("File is not a valid json")
