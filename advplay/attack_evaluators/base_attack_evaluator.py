"""Base class for attack evaluators, which turn an attack's output into metrics."""
from abc import ABC, abstractmethod

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

class BaseAttackEvaluator(ABC):
    """Registry-backed base class for computing metrics from an attack result.

    Subclasses register with ``attack_type`` (and optionally ``attack_subtype``); the
    orchestrator looks up the technique-specific evaluator first, then falls back to the
    ``(attack_type, None)`` evaluator.

    Attributes:
        registry: Maps ``(attack_type, attack_subtype)`` to the evaluator class.
    """

    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype: str = None, **kwargs):
        """Register the evaluator subclass under ``(attack_type, attack_subtype)``."""
        super().__init_subclass__(**kwargs)
        cls.attack_type = attack_type
        cls.attack_subtype = attack_subtype
        BaseAttackEvaluator.registry[(attack_type, attack_subtype)] = cls

    @classmethod
    def get(cls, attack_type: str, attack_subtype: str = None):
        """Return the evaluator for a technique, falling back to the category default.

        Args:
            attack_type: Attack category to look up.
            attack_subtype: Specific technique; ``None`` selects the category default.

        Returns:
            The matching evaluator class, or ``None`` if none is registered.
        """
        evaluator_cls = cls.registry.get((attack_type, attack_subtype))
        if evaluator_cls is None:
            evaluator_cls = cls.registry.get((attack_type, None))
        return evaluator_cls

    @abstractmethod
    def evaluate(self, context: BaseEvaluationContext):
        """Compute metrics for an attack.

        Args:
            context: Evaluation context carrying the attack's inputs and outputs.

        Returns:
            An ``(evaluation_results, models, visualization_context)`` tuple.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")
