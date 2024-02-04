from functools import lru_cache

import numpy as np

from kvfile.serialize import (
    EmbeddingSerializer,
    EmbeddingSerializerFactory, 
    NumpyToBytesEmbeddingsSerializer
)
from kvfile.storage.base import KVFile, KVFileBase, read_value


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
        serializer: str = "np.tobytes"
    ):
        self.storage_path = storage_path
        self.serializer = (
            EmbeddingSerializerFactory()
            .set_dim(emb_dim)
            .set_dtype(emb_dtype)
            .make_serializer(serializer)
        )
        
        self.__get_deserialize_cached_fn = lru_cache(maxsize=n_cached_values)(read_deserialize_value)

    def get_embedding(self, key: str) -> np.ndarray | None:
        return self.__get_deserialize_cached_fn(self.key2path(key), self.serializer)

    def set_embedding(self, key: str, embedding: np.ndarray):
        bytes_array = self.serializer.serialize(embedding=embedding)
        self.set(key, bytes_array)


def _embeddings_file_bulk_read(path, pos_in_file, emb_size) -> bytes | None:
    try:
        with open(path, "rb") as f:
            f.seek(emb_size * pos_in_file)
            return f.read(emb_size)
    except FileNotFoundError as e:
        return None


class EmbeddingsFileBulk(KVFileBase):
    dtype2size = {
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,

        np.float16: 2,
        np.float32: 4,
        np.float64: 8,
    }

    def __init__(
        self, 
        storage_path: str,
        emb_dim: int, 
        emb_dtype: type, 
        n_cached_values: int | None = None,
        n_embeddings_per_file: int = 1000
    ):
        self.storage_path = storage_path

        self.emb_dtype = emb_dtype
        self.serializer = NumpyToBytesEmbeddingsSerializer(self.emb_dtype)
        
        self.emb_dim = emb_dim
        self.emb_size = self.dtype2size[self.emb_dtype] * self.emb_dim

        self.n_embeddings_per_file = n_embeddings_per_file
        self.key2idx = {}

        if n_cached_values is None:
            self.__embeddings_file_bulk_read_cache = _embeddings_file_bulk_read
        else:
            self.__embeddings_file_bulk_read_cache = lru_cache(maxsize=n_cached_values)(_embeddings_file_bulk_read)
    
    def key2path(self, key: str) -> str:
        i = self.key2idx[key] // self.n_embeddings_per_file
        return self.storage_path + f"/{i}"
        
    def get(self, key: str) -> bytes | None:
        path = self.key2path(key)
        pos_in_file = self.key2idx[key] % self.n_embeddings_per_file
        return self.__embeddings_file_bulk_read_cache(path, pos_in_file=pos_in_file, emb_size=self.emb_size)

    def set(self, key: str, value: bytes):
        assert key not in self.key2idx
        
        self.key2idx[key] = len(self.key2idx)
        
        with open(self.key2path(key), "ab") as f:
            f.write(value)
    
    def get_embedding(self, key: str) -> np.ndarray | None:
        data = self.get(key=key)
        if data is None:
            return data
        
        return self.serializer.deserialize(data)

    def set_embedding(self, key: str, embedding: np.ndarray):
        bytes_array = self.serializer.serialize(embedding=embedding)
        self.set(key, bytes_array)
