import os
from typing import Annotated, Union
from pathlib import Path
import numpy as np

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths

class GoodwordsEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.GOODWORDS):
    source: Annotated[Union[int, str], AttackParam(type=(int, str), required=True, default=None,
                                                   help='Original class of the samples (the class to evade from)')]
    target: Annotated[Union[int, str], AttackParam(type=(int, str), required=True, default=None,
                                                   help='Class to evade into (the desired prediction)')]
    min_words: Annotated[int, AttackParam(type=int, required=False, default=0, help="Minimum words to add")]
    max_words: Annotated[int, AttackParam(type=int, required=False, default=40, help="Maximum words to add")]
    step: Annotated[int, AttackParam(type=int, required=False, default=5, help="Step size between words")]

    def execute(self):
        default_path = paths.MODELS
        if not Path(self.model_path).is_file():
            self.model_path = default_path / self.model_path
        loader_cls = BaseModelLoader.registry.get(self.training_framework)
        loader = loader_cls(self.model_path, self.model, self.training_configuration)
        model = loader.load()

        vectorizer = getattr(model, "vectorizer", None)
        if vectorizer is None:
            raise AttributeError(
                "Loaded model has no fitted vectorizer attached. "
                "Goodwords requires a model trained via NaiveBayesTrainer."
            )

        samples = np.asarray(self.samples_data).ravel().astype(str)

        labels_unique = np.unique(self.true_labels)
        label_map = {lbl: i for i, lbl in enumerate(labels_unique)}
        for name, value in (("source", self.source), ("target", self.target)):
            if value not in label_map:
                raise ValueError(f"{name} '{value}' not found in dataset labels {list(label_map.keys())}")

        source_idx = int(np.where(model.classes_ == label_map[self.source])[0][0])
        target_idx = int(np.where(model.classes_ == label_map[self.target])[0][0])

        feature_names = vectorizer.get_feature_names_out()
        source_log_probs = model.feature_log_prob_[source_idx]
        target_log_probs = model.feature_log_prob_[target_idx]

        goodness_scores = []
        for i, word in enumerate(feature_names):
            source_prob = np.exp(source_log_probs[i])
            target_prob = np.exp(target_log_probs[i])
            goodness = target_prob / (source_prob + 1e-10)
            goodness_scores.append((word, goodness))

        goodness_scores.sort(key=lambda x: x[1], reverse=True)

        word_counts = list(range(self.min_words, self.max_words + 1, self.step))
        if not word_counts or word_counts[-1] != self.max_words:
            word_counts.append(self.max_words)

        rates_per_count = []
        perturbed_samples = samples.copy()

        for num_words in word_counts:
            selected_words = [w for w, _ in goodness_scores[:num_words]]
            suffix = (" " + " ".join(selected_words)) if selected_words else ""
            augmented = np.array([s + suffix for s in samples])

            preds = model.predict(vectorizer.transform(augmented))
            evaded = int(np.sum(preds == target_idx))
            evasion_rate = (evaded / len(samples)) * 100

            rates_per_count.append({
                "num_words": num_words,
                "evaded": evaded,
                "evasion_rate": evasion_rate,
            })
            perturbed_samples = augmented

        context = EvasionEvaluationContext(
            self.model, None, None, self.training_framework,
            self.training_configuration, self.model_path,
            self.samples_data, perturbed_samples, self.target_label,
        )

        if self.dataset is not None:
            full_perturbed = np.insert(perturbed_samples.reshape(-1, 1), self.label_column,
                                       self.true_labels, axis=1)
            save_data = full_perturbed
            save_source_type = self.dataset.source_type
            save_metadata = self.dataset.metadata
        else:
            save_data = perturbed_samples
            save_source_type = self.samples.source_type
            save_metadata = self.samples.metadata

        dataset_name = save_metadata["dataset_name"]
        dataset_path = paths.DATASETS / 'perturbed_datasets' / dataset_name
        perturbed_dataset_path = paths.DATASETS / 'perturbed_datasets' / f"{dataset_name}_perturbed"
        os.makedirs(dataset_path.parent, exist_ok=True)

        loaded_dataset = LoadedDataset(save_data, save_source_type, save_metadata)
        datasets = [(loaded_dataset, str(perturbed_dataset_path))]

        attack_results = {
            "attack": self.attack_type,
            "technique": self.attack_subtype,
            "source": self.source,
            "target": self.target,
            "rates_per_word_count": rates_per_count,
            "original_dataset_path": str(dataset_path),
            "perturbed_dataset_path": str(perturbed_dataset_path),
        }

        return attack_results, context, datasets
