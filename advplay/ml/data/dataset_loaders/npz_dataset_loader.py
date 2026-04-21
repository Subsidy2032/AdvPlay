import numpy as np
import os

from advplay.ml.data.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.variables import dataset_formats

class NPZDatasetLoader(BaseDatasetLoader, source_type=dataset_formats.NPZ):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> LoadedDataset:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"NPZ file not found: {self.path}")

        dataset_file = np.load(self.path)

        keys = list(dataset_file.files)
        shapes = {k: dataset_file[k].shape for k in keys}

        reshaped = {
            k: (dataset_file[k].reshape(len(dataset_file[k]), -1)
                if dataset_file[k].ndim > 1
                else dataset_file[k].reshape(-1, 1))
            for k in keys
        }

        row_groups = {}
        for k in keys:
            row_groups.setdefault(reshaped[k].shape[0], []).append(k)

        if len(row_groups) == 1:
            dataset = np.hstack([reshaped[k] for k in keys])
            reference_keys = keys
        else:
            ordered_row_counts = sorted(row_groups.keys(), reverse=True)
            column_signature = None
            stacked_groups = []
            for row_count in ordered_row_counts:
                group_keys = row_groups[row_count]
                group_signature = tuple(reshaped[k].shape[1] for k in group_keys)
                if column_signature is None:
                    column_signature = group_signature
                elif group_signature != column_signature:
                    raise ValueError(
                        f"NPZ arrays cannot be combined: group with {row_count} rows has "
                        f"column shapes {group_signature}, expected {column_signature}"
                    )
                stacked_groups.append(np.hstack([reshaped[k] for k in group_keys]))
            dataset = np.vstack(stacked_groups)
            reference_keys = row_groups[ordered_row_counts[0]]

        key_columns = {}
        col_idx = 0
        for k in reference_keys:
            width = reshaped[k].shape[1]
            key_columns[k] = list(range(col_idx, col_idx + width))
            col_idx += width
        for group_keys in row_groups.values():
            for k, ref_k in zip(group_keys, reference_keys):
                key_columns.setdefault(k, key_columns[ref_k])

        metadata = {
            "keys": keys,
            "shapes": shapes,
            "key_columns": key_columns,
            "dataset_name": self.dataset_name,
            "dataset_path": self.path,
        }

        return LoadedDataset(dataset, source_type=self.source_type, metadata=metadata)
