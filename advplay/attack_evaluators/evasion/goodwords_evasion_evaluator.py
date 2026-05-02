from pathlib import Path
import numpy as np

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.goodwords_evasion_evaluation_context import GoodwordsEvasionEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.visualization.contexts.goodwords_evasion_visualization_context import GoodwordsEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


NUM_EXAMPLE_MESSAGES = 8
NUM_REPRESENTATIVE_COUNTS = 4


class GoodwordsEvasionEvaluator(BaseAttackEvaluator,
                                attack_type=available_attacks.EVASION,
                                attack_subtype=evasion_techniques.GOODWORDS):
    def evaluate(self, context: GoodwordsEvasionEvaluationContext):
        model_name = context.model
        training_framework = context.training_framework
        training_configuration = context.training_configuration
        model_path = context.model_path
        samples_data = context.samples_data
        perturbed_samples = context.perturbed_samples

        default_path = paths.MODELS
        if not Path(model_path).is_file():
            model_path = default_path / model_path
        loader_cls = BaseModelLoader.registry.get(training_framework)
        loader = loader_cls(model_path, model_name, training_configuration)
        model = loader.load()
        vectorizer = getattr(model, "vectorizer", None)

        evaluator_cls = BaseEvaluator.get(training_framework, model_name)
        evaluator = evaluator_cls(model)

        original_predictions = evaluator.predict(samples_data)
        perturbed_predictions = evaluator.predict(perturbed_samples)
        num_mispredictions = int(np.sum(np.asarray(original_predictions) != np.asarray(perturbed_predictions)))
        num_samples = int(len(np.asarray(original_predictions)))
        percentage_mispredicted = (num_mispredictions / num_samples) * 100 if num_samples else 0.0

        evaluation_results = {
            "original_predictions": original_predictions,
            "perturbed_predictions": perturbed_predictions,
            "num_mispredictions": num_mispredictions,
            "num_samples": num_samples,
            "percentage_mispredicted": percentage_mispredicted,
            "rates_per_word_count": context.rates_per_count,
            "top_word_contributions": [
                {"word": w, "avg_source_prob_reduction_pct": pct}
                for w, pct in context.word_contributions[:20]
            ],
        }

        example_messages, per_message_probs, representative_counts = self._compute_per_message_probabilities(
            model, vectorizer, context,
        )

        visualization_context = GoodwordsEvasionVisualizationContext(
            base_accuracy=None,
            source=context.source,
            target=context.target,
            word_counts=list(context.word_counts),
            evasion_rates=[r["evasion_rate"] for r in context.rates_per_count],
            top_word_contributions=context.word_contributions[:20],
            example_messages=example_messages,
            representative_word_counts=representative_counts,
            per_message_source_probs=per_message_probs,
        )

        print("Goodwords evasion evaluation summary:")
        print(f"Source: {context.source}  Target: {context.target}")
        for rate in context.rates_per_count:
            denom = rate.get("n_source_samples", num_samples)
            print(f"  +{rate['num_words']:>3} words → evasion {rate['evasion_rate']:5.1f}% "
                  f"({rate['evaded']} of {denom} {context.source} samples now predicted as {context.target})")
        if context.word_contributions:
            print("Top contributing words (avg source-prob reduction):")
            for w, pct in context.word_contributions[:5]:
                print(f"  {w:<20s} {pct:6.2f}%")

        return evaluation_results, [], visualization_context

    @staticmethod
    def _compute_per_message_probabilities(model, vectorizer, context):
        if vectorizer is None:
            return [], [], []

        samples = np.asarray(context.samples_data).ravel().astype(str)
        true_labels = np.asarray(context.true_labels)
        source_mask = true_labels == context.source
        source_messages = samples[source_mask]
        if len(source_messages) == 0:
            return [], [], []

        n_examples = min(NUM_EXAMPLE_MESSAGES, len(source_messages))
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(source_messages), size=n_examples, replace=False)
        examples = list(source_messages[chosen])

        word_counts = list(context.word_counts)
        if len(word_counts) <= NUM_REPRESENTATIVE_COUNTS:
            representative_counts = word_counts
        else:
            indices = np.linspace(0, len(word_counts) - 1, NUM_REPRESENTATIVE_COUNTS).round().astype(int)
            seen = set()
            representative_counts = []
            for i in indices:
                count = word_counts[int(i)]
                if count not in seen:
                    seen.add(count)
                    representative_counts.append(count)

        per_message_probs = []
        for msg in examples:
            row = []
            for k in representative_counts:
                suffix = (" " + " ".join(context.ranked_goodwords[:k])) if k else ""
                augmented = np.array([msg + suffix])
                proba = model.predict_proba(vectorizer.transform(augmented))[0, context.source_class_idx]
                row.append(float(proba))
            per_message_probs.append(row)

        return examples, per_message_probs, representative_counts
