import html
import re
import unicodedata

import numpy as np

from advplay.ml.data.preprocessors.base_preprocessor import BasePreprocessor
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset


_WHITESPACE_RE = re.compile(r"\s+")
_KEEP_CATEGORIES = {
    "Lu", "Ll", "Lt", "Lm", "Lo",
    "Nd", "Nl", "No",
    "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
    "Sc",
    "Zs",
}


def _clean_string(value: str) -> str:
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("+", " ")
    value = "".join(ch for ch in value if unicodedata.category(ch) in _KEEP_CATEGORIES)
    value = value.lower()
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def _is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _is_text_column(column: np.ndarray) -> bool:
    has_string = False
    for value in column:
        if _is_null(value):
            continue
        if not isinstance(value, str):
            return False
        has_string = True
    return has_string


class TextCleaner(BasePreprocessor, name="text_cleaner"):
    def apply(self, dataset: LoadedDataset) -> LoadedDataset:
        data = dataset.data
        was_1d = data.ndim == 1
        if was_1d:
            data = data.reshape(-1, 1)

        data = data.copy()
        text_columns = [i for i in range(data.shape[1]) if _is_text_column(data[:, i])]

        for i in text_columns:
            data[:, i] = np.array(
                [_clean_string(v) if isinstance(v, str) else v for v in data[:, i]],
                dtype=object,
            )

        if text_columns:
            keep_mask = np.ones(data.shape[0], dtype=bool)
            for i in text_columns:
                empty = np.array(
                    [not isinstance(v, str) or v == "" for v in data[:, i]]
                )
                keep_mask &= ~empty
            data = data[keep_mask]

        if data.shape[0] > 0:
            seen = set()
            keep_indices = []
            for i, row in enumerate(data):
                key = tuple(row.tolist()) if data.ndim > 1 else (row,)
                if key in seen:
                    continue
                seen.add(key)
                keep_indices.append(i)
            data = data[keep_indices]

        if was_1d:
            data = data.ravel()

        return LoadedDataset(data=data, source_type=dataset.source_type, metadata=dataset.metadata)
