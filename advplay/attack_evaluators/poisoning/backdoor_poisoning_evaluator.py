import numpy as np
from sklearn.metrics import confusion_matrix

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.backdoor_poisoning_evaluation_context import BackdoorPoisoningEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.ops.trainers.base_trainer import BaseTrainer
from advplay.visualization.contexts.backdoor_poisoning_visualization_context import BackdoorPoisoningVisualizationContext
from advplay.variables import available_attacks, poisoning_techniques
from advplay import paths
from advplay.utils import load_files


class BackdoorPoisoningEvaluator(BaseAttackEvaluator,
                                 attack_type=available_attacks.POISONING,
                                 attack_subtype=poisoning_techniques.BACKDOOR):
    def evaluate(self, context: BackdoorPoisoningEvaluationContext):
        training_framework = context.training_framework
        training_configuration = context.training_configuration
        if isinstance(training_configuration, str):
            default_path = paths.CONFIGS / training_framework
            training_configuration = load_files.load_json(default_path, training_configuration)

        evaluator_cls = BaseEvaluator.registry.get(training_framework)
        trainer_cls = BaseTrainer.registry.get((training_framework, context.model))

        X_test = context.X_test
        y_test = np.asarray(context.y_test).ravel()
        X_test_triggered = context.X_test_triggered
        target_label = context.target_label
        source_label = context.source_label
        labels_unique = context.labels
        label_indices = np.arange(len(labels_unique))

        source_mask_test = (
            (y_test == source_label) if source_label is not None else (y_test != target_label)
        )
        non_source_mask_test = ~source_mask_test & (y_test != target_label)

        base_trainer = trainer_cls(
            context.clean_dataset["X_train"],
            context.clean_dataset["y_train"],
            training_configuration,
        )
        base_model = base_trainer.train()
        base_evaluator = evaluator_cls(base_model)

        base_clean_preds = base_evaluator.predict(X_test)
        base_triggered_preds = base_evaluator.predict(X_test_triggered)
        base_clean_acc = float(np.mean(base_clean_preds == y_test))
        base_clean_cm = confusion_matrix(y_test, base_clean_preds, labels=label_indices)
        base_triggered_cm = confusion_matrix(y_test, base_triggered_preds, labels=label_indices)
        base_asr = _asr(base_triggered_preds, y_test, source_mask_test, target_label)
        base_triggered_non_source_acc = _masked_accuracy(base_triggered_preds, y_test, non_source_mask_test)

        per_class_asr_base = _per_class_asr(
            base_triggered_preds, y_test, target_label, labels_unique
        )

        models = [(training_framework, base_model, f"{context.model_name}_clean")]

        poisoning_results = []
        best = {"asr": -1.0, "portion": None, "model": None}

        for portion, data in context.poisoned_datasets.items():
            trainer = trainer_cls(data["X_train"], data["y_train"], training_configuration)
            model = trainer.train()
            evaluator = evaluator_cls(model)

            clean_preds = evaluator.predict(X_test)
            triggered_preds = evaluator.predict(X_test_triggered)

            clean_acc = float(np.mean(clean_preds == y_test))
            triggered_non_source_acc = _masked_accuracy(triggered_preds, y_test, non_source_mask_test)
            asr = _asr(triggered_preds, y_test, source_mask_test, target_label)

            clean_cm = confusion_matrix(y_test, clean_preds, labels=label_indices)
            triggered_cm = confusion_matrix(y_test, triggered_preds, labels=label_indices)

            per_class_asr = _per_class_asr(triggered_preds, y_test, target_label, labels_unique)

            poisoning_results.append({
                "portion": portion,
                "n_samples_poisoned": data["n_samples_poisoned"],
                "clean_accuracy": clean_acc,
                "triggered_non_source_accuracy": triggered_non_source_acc,
                "asr": asr,
                "clean_confusion_matrix": clean_cm,
                "triggered_confusion_matrix": triggered_cm,
                "per_class_asr": per_class_asr,
            })

            if asr > best["asr"]:
                best = {"asr": asr, "portion": portion, "model": model,
                        "clean_accuracy": clean_acc,
                        "triggered_non_source_accuracy": triggered_non_source_acc}

        if best["model"] is not None:
            models.append((training_framework, best["model"], f"{context.model_name}_backdoored"))

        evaluation_results = {
            "trigger": context.trigger,
            "source_class": context.source_class,
            "target_class": context.target_class,
            "base_clean_accuracy": base_clean_acc,
            "base_triggered_non_source_accuracy": base_triggered_non_source_acc,
            "base_asr": base_asr,
            "poisoning_results": [
                {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
                for r in poisoning_results
            ],
            "best": {k: v for k, v in best.items() if k != "model"},
        }

        print("Backdoor poisoning evaluation summary:")
        print(f"Trigger: {context.trigger}")
        print(f"Source class: {context.source_class}  Target class: {context.target_class}")
        print(f"Baseline clean accuracy:                {base_clean_acc:.4f}")
        print(f"Baseline trigger-only non-source acc:   {base_triggered_non_source_acc:.4f}")
        print(f"Baseline ASR (no poisoning):            {base_asr:.4f}")
        for r in poisoning_results:
            print(
                f"  portion={r['portion']*100:5.1f}%  "
                f"clean_acc={r['clean_accuracy']:.4f}  "
                f"non_src_triggered_acc={r['triggered_non_source_accuracy']:.4f}  "
                f"ASR={r['asr']:.4f}"
            )
        if best["portion"] is not None:
            print(f"Best ASR {best['asr']:.4f} at portion {best['portion']*100:.1f}% "
                  f"(clean acc {best['clean_accuracy']:.4f})")

        portions = [0.0] + [r["portion"] for r in poisoning_results]
        percentages = [p * 100 for p in portions]
        n_samples = [0] + [r["n_samples_poisoned"] for r in poisoning_results]
        clean_accuracies = [base_clean_acc] + [r["clean_accuracy"] for r in poisoning_results]
        triggered_non_source_accuracies = [base_triggered_non_source_acc] + \
            [r["triggered_non_source_accuracy"] for r in poisoning_results]
        asrs = [base_asr] + [r["asr"] for r in poisoning_results]
        clean_cms = [base_clean_cm] + [r["clean_confusion_matrix"] for r in poisoning_results]
        triggered_cms = [base_triggered_cm] + [r["triggered_confusion_matrix"] for r in poisoning_results]
        per_class_asr_by_portion = [per_class_asr_base] + [r["per_class_asr"] for r in poisoning_results]
        non_target_labels = [lbl for lbl in labels_unique if lbl != context.target_class]

        poisoned_model = best["model"] if best["model"] is not None else base_model
        poisoned_evaluator = evaluator_cls(poisoned_model)

        example_clean = np.expand_dims(context.example_clean, axis=0)
        example_triggered = np.expand_dims(context.example_triggered, axis=0)
        example_clean_pred_base = _pred_label(base_evaluator, example_clean, labels_unique)
        example_triggered_pred_base = _pred_label(base_evaluator, example_triggered, labels_unique)
        example_clean_pred_poisoned = _pred_label(poisoned_evaluator, example_clean, labels_unique)
        example_triggered_pred_poisoned = _pred_label(poisoned_evaluator, example_triggered, labels_unique)

        visualization_context = BackdoorPoisoningVisualizationContext(
            base_accuracy=base_clean_acc,
            base_clean_confusion_matrix=base_clean_cm,
            base_triggered_confusion_matrix=base_triggered_cm,
            base_asr=base_asr,
            base_triggered_non_source_accuracy=base_triggered_non_source_acc,
            source_class=context.source_class,
            target_class=context.target_class,
            source_label=source_label,
            target_label=target_label,
            labels=labels_unique,
            trigger=context.trigger,
            portions_poisoned=portions,
            percentages_poisoned=percentages,
            n_samples_poisoned=n_samples,
            clean_accuracies=clean_accuracies,
            triggered_non_source_accuracies=triggered_non_source_accuracies,
            asrs=asrs,
            clean_confusion_matrices=clean_cms,
            triggered_confusion_matrices=triggered_cms,
            per_class_asr_by_portion=per_class_asr_by_portion,
            non_target_class_labels=non_target_labels,
            example_clean=context.example_clean,
            example_triggered=context.example_triggered,
            example_true_label=context.example_true_label,
            example_clean_prediction_base=example_clean_pred_base,
            example_triggered_prediction_base=example_triggered_pred_base,
            example_clean_prediction_poisoned=example_clean_pred_poisoned,
            example_triggered_prediction_poisoned=example_triggered_pred_poisoned,
        )

        return evaluation_results, models, visualization_context


def _asr(predictions, y_true, source_mask, target_label):
    if not np.any(source_mask):
        return 0.0
    source_preds = predictions[source_mask]
    source_true = y_true[source_mask]
    eligible = source_true != target_label
    if not np.any(eligible):
        return 0.0
    return float(np.mean(source_preds[eligible] == target_label))


def _masked_accuracy(predictions, y_true, mask):
    if not np.any(mask):
        return float("nan")
    return float(np.mean(predictions[mask] == y_true[mask]))


def _per_class_asr(predictions, y_true, target_label, labels_unique):
    per_class = {}
    for idx, original_label in enumerate(labels_unique):
        if idx == target_label:
            continue
        mask = y_true == idx
        if not np.any(mask):
            per_class[original_label] = float("nan")
            continue
        per_class[original_label] = float(np.mean(predictions[mask] == target_label))
    return per_class


def _pred_label(evaluator, X, labels_unique):
    pred = evaluator.predict(X)
    pred_idx = int(np.asarray(pred).ravel()[0])
    if 0 <= pred_idx < len(labels_unique):
        return labels_unique[pred_idx]
    return pred_idx
