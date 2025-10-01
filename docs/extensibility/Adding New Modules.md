Adding new modules is simple.

Note: Do not forget to remove the `# TEMPLATE FILE` comment from the top of template files if you copy them. Otherwise, the new module will not be recognized.

## Adding a New Attack

To add support for a new attack, follow the following steps:

1. Add a new folder named after the attack in the `advplay/attacks` folder.
2. Add a new Python file with a class following the template in the `advplay/attacks/template_attack_classes/template_attack.py` file.

To add a new technique for an existing attack, follow the template in the `advplay/attacks/template_attack_classes/sub_template_attack.py` file.

## Adding a New Visualization Module

You can follow the templates under the `advplay/visualization/template_visulization_class` folder in a similar way to adding a new attack.

## Adding Other Modules

Under model_ops, it is possible to add support for additional trainers, LLM platforms, or other model and data related processes. You can follow existing modules as a reference.
