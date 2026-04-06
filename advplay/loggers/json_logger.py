import json
import numpy as np

from advplay.loggers.base_logger import BaseLogger

class JsonLogger(BaseLogger):
    def log(self, results: dict):
        self.append_log_entry(self.location, results)

    def append_log_entry(self, log_file_path, log_entry):
        if not str(log_file_path).endswith(".log"):
            log_file_path += ".log"
            
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
                if not isinstance(existing_logs, list):
                    existing_logs = []
        except (FileNotFoundError, json.JSONDecodeError):
            existing_logs = []

        existing_logs.append(log_entry)

        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, ensure_ascii=False, indent=2, default=self.convert_numpy)


    def convert_numpy(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj