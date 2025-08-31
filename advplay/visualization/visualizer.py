import json

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay import paths
from advplay.utils import load_files

def visualizer(attack_type, **kwargs):
    log_file = kwargs.get("log_filename")

    default_path = paths.ATTACK_LOGS / attack_type
    if isinstance(log_file, str):
        log_file = load_files.load_json(default_path, log_file)

    if not isinstance(log_file[0], dict):
        raise TypeError(f"log file must be a JSON object (dict), got {type(log_file).__name__}")

    technique = log_file[0].get("technique")
    key = (attack_type, technique)
    visualizer_cls = BaseVisualizer.registry.get(key)
    if visualizer_cls is None:
        raise ValueError(f"Unsupported attack type and subtype: {key}")

    print(f"Visualizing the '{attack_type}' attack")
    visualizer = visualizer_cls(log_file[0], **kwargs)
    visualizer.visualize()
