import os
import importlib
from pathlib import Path

from advplay.paths import TEMPLATE_BUILDERS, PROJECT_ROOT, ATTACKS, HANDLERS

def load_required_classes():
    base_paths = [TEMPLATE_BUILDERS, ATTACKS, HANDLERS]

    for base_path in base_paths:
        for dirpath, _, filenames in os.walk(base_path):
            if Path(dirpath) == base_path:
                continue

            for filename in filenames:
                if filename.endswith(".py") and not filename.startswith("__"):
                    full_path = Path(dirpath) / filename
                    rel_path = full_path.relative_to(PROJECT_ROOT)
                    module_parts = rel_path.with_suffix("").parts
                    module_name = ".".join(module_parts)

                    try:
                        mod = importlib.import_module(module_name)

                    except Exception as e:
                        print(f"Failed to import {module_name}: {e}")
                        continue
