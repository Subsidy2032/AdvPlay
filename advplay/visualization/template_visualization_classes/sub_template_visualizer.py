# TEMPLATE FILE
from advplay.visualization.template_visualization_classes.template_visualizer import TemplateVisualizer
from advplay.variables import available_attacks, poisoning_techniques

class SubTemplateVisualizer(TemplateVisualizer, attack_type='template', attack_subtype='sub'):
    def visualize(self):
        pass