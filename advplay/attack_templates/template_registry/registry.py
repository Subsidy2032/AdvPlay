TEMPLATE_BUILDERS = {}

def register_template_builder(template_type: str):
    def decorator(cls):
        TEMPLATE_BUILDERS[template_type] = cls
        return cls
    return decorator

def define_template(template_type: str, **kwargs):
    builder_cls = TEMPLATE_BUILDERS.get(template_type)
    if builder_cls is None:
        raise ValueError(f"Unsupported template type: {template_type}")
    builder = builder_cls(**kwargs)
    builder.build()
