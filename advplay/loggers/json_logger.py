"""JSON logger that appends attack results to a ``.log`` file as a JSON array."""
import json
import numpy as np

from advplay.loggers.base_logger import BaseLogger

class JsonLogger(BaseLogger):
    """Append attack results to a ``.log`` file holding a JSON array of entries."""

    def log(self, results: dict):
        """Append ``results`` to the log file at ``self.location``."""
        self.append_log_entry(self.location, results)
        print(f"Log results are saved to the {self.location}.log file\n")

    def append_log_entry(self, log_file_path, log_entry):
        """Append one entry to a JSON-array log file, creating or resetting it as needed."""
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
        """Convert numpy scalars/arrays to JSON-serializable Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj