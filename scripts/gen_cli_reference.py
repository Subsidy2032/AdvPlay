"""Generate the CLI reference pages at build time.

Run automatically by the `mkdocs-gen-files` plugin (see ``mkdocs.yml``). It walks
``BaseAttack.registry`` the same way ``run.py`` builds its argument parser, so the
generated reference always matches the real CLI. The per-parameter descriptions come
straight from the ``help=`` text declared on each ``AttackParam``/``TemplateParam`` —
there is no second copy of that text to keep in sync.

Nothing is written to disk in the repo: ``mkdocs_gen_files.open`` writes into the built
site only.
"""
import os
import re
import sys

import mkdocs_gen_files

# mkdocs runs from the repo root (where mkdocs.yml lives), but the `advplay` package is
# not installed, so make sure the root is importable before touching the registry.
sys.path.insert(0, os.getcwd())

from advplay.utils.load_classes import load_required_classes
from advplay.attacks.base_attack import BaseAttack

# Importing every module populates BaseAttack.registry (mirrors run.py:16).
load_required_classes()

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def render_type(t):
    if t is None:
        return ""
    if isinstance(t, tuple):
        return " or ".join(render_type(x) for x in t)
    name = getattr(t, "__name__", None)
    return f"`{name}`" if name else f"`{t}`"


def render_default(d):
    if d is None:
        return ""
    # model_name / log_filename default to datetime.now() captured at import time;
    # showing that timestamp in docs would be misleading.
    if isinstance(d, str) and _TIMESTAMP_RE.match(d):
        return "_current timestamp_"
    return f"`{d}`"


def render_choices(c):
    if callable(c):
        try:
            c = c()
        except Exception:
            return ""
    if not c:
        return ""
    return ", ".join(f"`{x}`" for x in c)


def render_help(h):
    if not h:
        return ""
    return h.replace("|", "\\|").replace("\n", " ").strip()


def params_table(params):
    if not params:
        return "_No parameters._"
    rows = [
        "| Flag | Type | Required | Default | Choices | Description |",
        "|------|------|----------|---------|---------|-------------|",
    ]
    for name, meta in params.items():
        flag = f"`--{name.replace('_', '-')}`"
        rows.append(
            f"| {flag} | {render_type(meta.type)} | "
            f"{'Yes' if meta.required else 'No'} | {render_default(meta.default)} | "
            f"{render_choices(meta.choices)} | {render_help(meta.help)} |"
        )
    return "\n".join(rows)


registry = BaseAttack.registry
categories = sorted({key[0] for key in registry})

# --- Overview page ---------------------------------------------------------------
overview = [
    "# CLI reference",
    "",
    "AdvPlay is driven entirely from the command line:",
    "",
    "```",
    "python3 run.py <command> ...",
    "```",
    "",
    "There are two commands:",
    "",
    "- **`save_template`** — write a reusable attack configuration to a JSON template.",
    "- **`attack`** — run an attack from a saved template.",
    "",
    "The usual flow is to save a template once, then reference it by name when running "
    "attacks. Every page below is generated from the attack registry, so it lists exactly "
    "the flags the CLI accepts.",
    "",
    "## Attack categories",
    "",
]
for category in categories:
    overview.append(f"- [`{category}`]({category}.md)")
overview.append("")

with mkdocs_gen_files.open("cli/index.md", "w") as f:
    f.write("\n".join(overview))

# --- Per-category pages ----------------------------------------------------------
nav_lines = ["* [Overview](index.md)"]

for category in categories:
    base_cls = registry[(category, None)]
    techniques = sorted(
        key[1] for key in registry if key[0] == category and key[1] is not None
    )

    page = [
        f"# `{category}` attacks",
        "",
        "## Save a template",
        "",
        "```",
        f"python3 run.py save_template {category} [--template-filename NAME] [options]",
        "```",
        "",
        f"List existing templates with `--list`, or inspect one with `--template NAME`. "
        f"The configuration fields are:",
        "",
        params_table(base_cls.TEMPLATE_PARAMETERS),
        "",
        "## Run an attack",
        "",
    ]

    if not techniques:
        page.append("_No techniques are registered for this category._")
    for technique in techniques:
        technique_cls = registry[(category, technique)]
        page += [
            f"### `{technique}`",
            "",
            "```",
            f"python3 run.py attack {category} {technique} --template NAME [options]",
            "```",
            "",
            params_table(technique_cls.ATTACK_PARAMETERS),
            "",
        ]

    with mkdocs_gen_files.open(f"cli/{category}.md", "w") as f:
        f.write("\n".join(page))

    nav_lines.append(f"* [{category}]({category}.md)")

with mkdocs_gen_files.open("cli/SUMMARY.md", "w") as f:
    f.write("\n".join(nav_lines) + "\n")
