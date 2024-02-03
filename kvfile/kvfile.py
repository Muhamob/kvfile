from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from kvfile.serialize import NumpyEmbeddingsSerializer, StructEmbeddingSerializer


class KVFileBase(ABC):
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

        self.get_fn = lru_cache(maxsize=self.n_cached_values)(read_value)

    def key2path(self, key: str) -> str:
        return key

    def get(self, key: str) -> bytes | None:
        return self.get_fn(self.storage_path + f"/{self.key2path(key)}")

    def set(self, key: str, value: bytes):
        with open(self.storage_path + f"/{key}", "wb") as f:
            f.write(value)


class EmbeddingsFile(KVFile):
    def __init__(
        self, 
        storage_path: str, 
        emb_dim: int,
        emb_dtype: type,
        n_cached_values: int | None = None,
        serializer: str = "struct"
    ):
        super().__init__(storage_path=storage_path, n_cached_values=n_cached_values)

        if serializer == "struct":
            self.serializer = StructEmbeddingSerializer(dim=emb_dim, dtype=emb_dtype)
        elif serializer == "numpy":
            self.serializer = NumpyEmbeddingsSerializer()
        else:
            raise TypeError(f"{serializer} serializer not found")

    def get_embedding(self, key: str) -> np.ndarray | None:
        bytes_array = self.get(key)
        
        if bytes_array is None:
            return None
        
        return self.serializer.deserialize(bytes_array)

    def set_embedding(self, key: str, embedding: np.ndarray):
        bytes_array = self.serializer.serialize(embedding=embedding)
        self.set(key, bytes_array)
