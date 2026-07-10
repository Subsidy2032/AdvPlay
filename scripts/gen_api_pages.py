"""Generate the Python API reference pages at build time.

Run automatically by the `mkdocs-gen-files` plugin (see ``mkdocs.yml``). For every module
under ``advplay/`` it emits a stub page containing an mkdocstrings ``:::`` directive;
mkdocstrings then renders each module's classes and functions from their docstrings.

mkdocstrings analyses the source statically (via Griffe), so this never imports torch,
ART or any other heavy runtime dependency.
"""
from pathlib import Path

import mkdocs_gen_files

_TEMPLATE_MARKER = "# TEMPLATE FILE"

nav = mkdocs_gen_files.Nav()
package = Path("advplay")

for path in sorted(package.rglob("*.py")):
    module_path = path.with_suffix("")          # e.g. advplay/attacks/base_attack
    doc_path = path.relative_to(package).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = tuple(module_path.parts)            # ("advplay", "attacks", "base_attack")

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("__"):
        continue

    # Skip template scaffold files, which load_required_classes also ignores at runtime.
    with path.open("r", encoding="utf-8") as fh:
        if _TEMPLATE_MARKER in fh.readline():
            continue

    identifier = ".".join(parts)
    # Drop the redundant top-level "advplay" node from the nav tree; keep the full,
    # importable identifier for the mkdocstrings directive.
    nav_key = parts[1:] or (parts[0],)
    nav[nav_key] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # No manual heading: mkdocstrings renders the module name as the page heading
        # (show_root_heading), so adding one here would duplicate it.
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/index.md", "w") as f:
    f.write(
        "# API reference\n\n"
        "These pages are generated from the source of the `advplay` package. Each module's\n"
        "classes and functions are shown with their signatures; descriptions appear where\n"
        "docstrings have been written (Google style), and coverage grows over time.\n\n"
        "Good entry points:\n\n"
        "- `advplay.attacks.base_attack` — the `BaseAttack` registry that every attack builds on.\n"
        "- `advplay.attack_evaluators.base_attack_evaluator` — metric computation.\n"
        "- `advplay.visualization.base_visualizer` — result plotting.\n"
        "- `advplay.orchestrators.full_pipeline_orchestrator` — wires attack → eval → log → viz.\n\n"
        "See the [Extending AdvPlay](../Extending AdvPlay/Extending AdvPlay.md) guide for how\n"
        "these base classes fit together.\n"
    )

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write("* [Overview](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
