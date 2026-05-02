from dataclasses import dataclass, field
from typing import Any, List, Tuple

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext


@dataclass
class GoodwordsEvasionVisualizationContext(BaseVisualizationContext):
    source: Any = None
    target: Any = None

    word_counts: List[int] = field(default_factory=list)
    evasion_rates: List[float] = field(default_factory=list)

    top_word_contributions: List[Tuple[str, float]] = field(default_factory=list)

    example_messages: List[str] = field(default_factory=list)
    representative_word_counts: List[int] = field(default_factory=list)
    per_message_source_probs: List[List[float]] = field(default_factory=list)
