from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase

def define_template(template_type: str, attack_type: str, **kwargs):
    builder_cls = TemplateBuilderBase.registry.get(attack_type).get(template_type)
    if builder_cls is None:
        raise ValueError(f"Unsupported template type: {template_type}")

    print(f"Creating a template for {attack_type} attack with type {template_type}")
    builder = builder_cls(attack_type, **kwargs)
    builder.build()
