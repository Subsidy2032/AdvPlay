import os
import json

from advplay.paths import TEMPLATES

def list_template_names(template_type: str):
    directory = TEMPLATES / template_type

    if directory.exists() and directory.is_dir():
        files = [
            filename.stem
            for filename in directory.iterdir()
            if filename.suffix == ".json" and filename.is_file()
        ]
    else:
        files = []

    print("Available templates:")
    for file in files:
        print(f" - {file}")

def list_template_contents(template_type: str, template_name: str):
    file_path = os.path.join(TEMPLATES / template_type, f"{template_name}.json")

    if not os.path.isfile(file_path):
        raise ValueError(f"Template '{template_name}' not found in '{TEMPLATES / template_type}'.")

    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

    print(f"{template_name}:")
    for key, value in data.items():
        print(f" - {key}: {value}")