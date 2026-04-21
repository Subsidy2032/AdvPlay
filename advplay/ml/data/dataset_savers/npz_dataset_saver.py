import numpy as np

from advplay.ml.data.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.variables import dataset_formats

class NPZDatasetSaver(BaseDatasetSaver, source_type=dataset_formats.NPZ):
    def save(self):
        target = self.path if self.path.suffix else self.path.with_suffix(".npz")
        keys, shapes = self.metadata["keys"], self.metadata["shapes"]
        key_columns = self.metadata.get("key_columns")

        if key_columns:
            arrays = self.split_by_groups(keys, shapes, key_columns)
        else:
            arrays = self.split_sequentially(keys, shapes)

        np.savez(str(target), **arrays)

    def split_by_groups(self, keys, shapes, key_columns):
        row_groups = {}
        for key in keys:
            row_count = int(shapes[key][0])
            row_groups.setdefault(row_count, []).append(key)

        ordered_row_counts = sorted(row_groups.keys(), reverse=True)
        expected_rows = sum(ordered_row_counts)

        if expected_rows != self.data.shape[0]:
            reference_keys = row_groups[ordered_row_counts[0]]
            arrays = {}
            for key in reference_keys:
                columns = key_columns[key]
                slice_ = self.data[:, columns]
                new_shape = (self.data.shape[0],) + tuple(shapes[key][1:])
                arrays[key] = slice_.reshape(new_shape)
            return arrays

        arrays = {}
        row_offset = 0
        for row_count in ordered_row_counts:
            group_keys = row_groups[row_count]
            for key in group_keys:
                columns = key_columns[key]
                slice_ = self.data[row_offset:row_offset + row_count, columns]
                arrays[key] = slice_.reshape(tuple(shapes[key]))
            row_offset += row_count

        return arrays

    def split_sequentially(self, keys, shapes):
        arrays, offset = {}, 0
        for key in keys:
            shape = tuple(shapes[key])
            cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            slice_ = self.data[:, offset:offset + cols]
            offset += cols
            arrays[key] = slice_.reshape(shape)
        return arrays