import os
import json

from advplay.paths import TEMPLATES

def list_template_names(template_type):
    files = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(TEMPLATES / template_type)
        if filename.endswith(".json") and os.path.isfile(os.path.join(TEMPLATES / template_type, filename))
    ]

    print("Available templates:")
    for file in files:
        print(f" - {file}")

def list_template_contents(template_type, template_name):
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