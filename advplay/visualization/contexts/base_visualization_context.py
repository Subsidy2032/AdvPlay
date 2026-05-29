from dataclasses import dataclass

from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset

@dataclass
class BaseVisualizationContext:
    base_accuracy: any

    def denormalize(self, denormalizers):
        pass

    @staticmethod
    def _denormalize_array(denormalizers, array, source_type=None, metadata=None):
        dataset = LoadedDataset(array, source_type=source_type, metadata=metadata or {})
        for denormalizer in denormalizers:
            dataset = denormalizer.apply(dataset)
        return dataset.data
