# TEMPLATE FILE
from abc import ABC
from datetime import datetime

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.variables import available_attacks
from advplay import paths

class TemplateVisualizer(BaseVisualizer, ABC, attack_type='template', attack_subtype=None):
    def __init__(self, log_file: dict, **kwargs):
        super().__init__(log_file, **kwargs)
        # Initialize values from the log file with log_file.get("value")

        self.directory_name = (paths.VISUALIZATIONS_RESULTS / available_attacks.POISONING /
                               kwargs.get('directory', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.directory_name.mkdir(parents=True, exist_ok=True)
