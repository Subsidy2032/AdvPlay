import json
from pathlib import Path

def load_json(default_path: str, config_input: str) -> dict:
    path = Path(config_input)

    if not path.suffix:
        for ext in (".json", ".log"):
            candidate = path.with_suffix(ext)
            if candidate.is_file():
                path = candidate
                break

            elif (Path(default_path) / candidate).is_file():
                path = Path(default_path) / candidate
                break

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_input} in {default_path}")

    try:
        with open(path, "r") as f:
            return json.load(f)

    except Exception as e:
        raise ValueError("File is not a valid json")
