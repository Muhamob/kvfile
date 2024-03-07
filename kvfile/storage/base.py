from abc import ABC, abstractmethod
from functools import lru_cache
import os

import numpy as np


class KVFileBase(ABC):
    dtype2size = {
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,

        np.float16: 2,
        np.float32: 4,
        np.float64: 8,
    }

    dtype2str = {
        np.int16: "np.int16",
        np.int32: "np.int32",
        np.int64: "np.int64",

        np.float16: "np.float16",
        np.float32: "np.float32",
        np.float64: "np.float64",
    }
    str2dtype = {v: k for k, v in dtype2str.items()}

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        pass

    @abstractmethod
    def set(self, key: str, value: bytes):
        pass


def read_value(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError as e:
        data = None

    return data


class KVFile(KVFileBase):
    def __init__(self, storage_path: str, n_cached_values: int | None = None):
        self.storage_path = storage_path
        self.n_cached_values = n_cached_values

        if n_cached_values is not None:
            self._get_fn = lru_cache(maxsize=self.n_cached_values)(read_value)
        else:
            self._get_fn = read_value
        self.cached = n_cached_values is not None

    def key2path(self, key: str) -> str:
        return self.storage_path + f"/{key}"
    
    def clear_cache(self):
        if self.cached:
            self._get_fn.cache_clear()

    def get(self, key: str) -> bytes | None:
        return self._get_fn(self.key2path(key))

    def set(self, key: str, value: bytes):
        if os.path.exists(self.key2path(key)):
            self.clear_cache()
        
        with open(self.key2path(key), "wb") as f:
            f.write(value)
