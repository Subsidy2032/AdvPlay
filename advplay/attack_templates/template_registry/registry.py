from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase

def define_template(template_type: str, attack_type, **kwargs):
    builder_cls = TemplateBuilderBase.registry.get(template_type)
    if builder_cls is None:
        raise ValueError(f"Unsupported template type: {template_type}")
    builder = builder_cls(attack_type, **kwargs)
    builder.build()
