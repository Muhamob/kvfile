from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from kvfile.serialize import (
    EmbeddingSerializer,
    NumpySaveEmbeddingsSerializer, 
    StructEmbeddingSerializer,
    NumpyToBytesEmbeddingsSerializer
)


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
        return self.storage_path + f"/{key}"

    def get(self, key: str) -> bytes | None:
        return self.get_fn(self.key2path(key))

    def set(self, key: str, value: bytes):
        with open(self.key2path(key), "wb") as f:
            f.write(value)


def read_deserialize_value(path: str, serializer: EmbeddingSerializer) -> np.ndarray | None:
    data = read_value(path=path)
    if data is None:
        return data
    
    return serializer.deserialize(data)


class EmbeddingsFile(KVFile):
    def __init__(
        self, 
        storage_path: str, 
        emb_dim: int,
        emb_dtype: type,
        n_cached_values: int | None = None,
        serializer: str = "numpybuffer"
    ):
        self.storage_path = storage_path

        if serializer == "struct":
            self.serializer = StructEmbeddingSerializer(dim=emb_dim, dtype=emb_dtype)
        elif serializer == "numpy":
            self.serializer = NumpySaveEmbeddingsSerializer()
        elif serializer == "numpybuffer":
            self.serializer = NumpyToBytesEmbeddingsSerializer(emb_dtype)
        else:
            raise TypeError(f"{serializer} serializer not found")
        
        self.__get_deserialize_cached_fn = lru_cache(maxsize=n_cached_values)(read_deserialize_value)

    def get_embedding(self, key: str) -> np.ndarray | None:
        return self.__get_deserialize_cached_fn(self.key2path(key), self.serializer)

    def set_embedding(self, key: str, embedding: np.ndarray):
        bytes_array = self.serializer.serialize(embedding=embedding)
        self.set(key, bytes_array)
