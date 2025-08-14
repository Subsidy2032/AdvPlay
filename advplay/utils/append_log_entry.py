import json


def append_log_entry(log_file_path, log_entry):
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            existing_logs = json.load(f)
            if not isinstance(existing_logs, list):
                existing_logs = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_logs = []

    existing_logs.append(log_entry)

    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=2)
