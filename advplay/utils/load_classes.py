import os
import importlib
from pathlib import Path

from advplay import paths

TEMPLATE_MARKER = "# TEMPLATE FILE"
def load_required_classes():
    base_path = paths.ADVPLAY

    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(".py") and not filename.startswith("__"):
                full_path = Path(dirpath) / filename

                with open(full_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    if TEMPLATE_MARKER in first_line:
                        continue

                rel_path = full_path.relative_to(paths.PROJECT_ROOT)
                module_parts = rel_path.with_suffix("").parts
                module_name = ".".join(module_parts)

                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    print(f"Failed to import {module_name}: {e}")