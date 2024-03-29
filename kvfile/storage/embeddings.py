from functools import lru_cache
from itertools import groupby
import json
import os
from typing import Sequence

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
        
        if n_cached_values is not None:
            self.__get_deserialize_cached_fn = lru_cache(maxsize=n_cached_values)(read_deserialize_value)
        else:
            self.__get_deserialize_cached_fn = read_deserialize_value
        self.cached = n_cached_values is not None

        self.emb_dtype = emb_dtype
        self.emb_dim = emb_dim
        self.serializer_name = serializer

    def clear_cache(self):
        if self.cached:
            self.__get_deserialize_cached_fn.cache_clear()

    def get_embedding(self, key: str) -> np.ndarray | None:
        return self.__get_deserialize_cached_fn(self.key2path(key), self.serializer)

    def set_embedding(self, key: str, embedding: np.ndarray):
        bytes_array = self.serializer.serialize(embedding=embedding)
        self.set(key, bytes_array)

    def get(self, key: str) -> bytes | None:
        return read_value(self.key2path(key))

    def set(self, key: str, value: bytes):
        if os.path.exists(self.key2path(key)):
            self.clear_cache()
        
        with open(self.key2path(key), "wb") as f:
            f.write(value)

    def dump(self):
        with open(self.storage_path + "/metadata.json", "w") as f:
            json.dump({
                "emb_dtype": self.dtype2str[self.emb_dtype],
                "emb_dim": self.emb_dim,
                "n_cached_values": self.n_cached_values,
                "serializer": self.serializer_name
            }, f)

    @classmethod
    def load(cls, storage_path: str):
        with open(storage_path + "/metadata.json", "r") as f:
            data = json.load(f)

        _cls = cls(
            storage_path=storage_path, 
            emb_dim=data["emb_dim"], 
            emb_dtype=cls.str2dtype[data["emb_dtype"]], 
            n_cached_values=data["n_cached_values"], 
            serializer=data["serializer"]
        )

        return _cls


def _embedding_file_bulk_read(path, position_in_file, emb_size) -> bytes | None:
    try:
        with open(path, "rb") as f:
            f.seek(emb_size * position_in_file)
            return f.read(emb_size)
    except FileNotFoundError as e:
        return None


class EmbeddingsFileBulk(KVFileBase):
    def __init__(
        self, 
        storage_path: str,
        emb_dim: int, 
        emb_dtype: type, 
        n_cached_values: int | None = None,
        n_embeddings_per_file: int = 1000
    ):
        """
        Key Value on Disk storage with n_embeddings_per_file embeddings (pair of key, value) in one file.
        """
        self.storage_path = storage_path

        self.emb_dtype = emb_dtype
        self.serializer = NumpyToBytesEmbeddingsSerializer(self.emb_dtype)
        
        self.emb_dim = emb_dim
        self.emb_size = self.dtype2size[self.emb_dtype] * self.emb_dim

        self.n_embeddings_per_file = n_embeddings_per_file
        self.key2idx = {}

        self.n_cached_values = n_cached_values
        if n_cached_values is None:
            self.__embedding_file_bulk_read_cache = _embedding_file_bulk_read
        else:
            self.__embedding_file_bulk_read_cache = lru_cache(maxsize=n_cached_values)(_embedding_file_bulk_read)
        self.cached = n_cached_values is not None
    
    def key2path(self, key: str) -> str:
        i = self.key2idx[key] // self.n_embeddings_per_file
        return self.storage_path + f"/{i}.data"
    
    def clear_cache(self):
        if self.cached:
            self.__embedding_file_bulk_read_cache.cache_clear()
    
    def _get_pos_in_file(self, key):
        return self.key2idx[key] % self.n_embeddings_per_file
        
    def get(self, key: str) -> bytes | None:
        path = self.key2path(key)
        pos_in_file = self._get_pos_in_file(key)
        return self.__embedding_file_bulk_read_cache(path, position_in_file=pos_in_file, emb_size=self.emb_size)

    def dump(self):
        with open(self.storage_path + "/key2idx.json", "w") as f:
            json.dump(self.key2idx, f)

        with open(self.storage_path + "/metadata.json", "w") as f:
            json.dump({
                "emb_dtype": self.dtype2str[self.emb_dtype],
                "emb_dim": self.emb_dim,
                "n_cached_values": self.n_cached_values,
                "n_embeddings_per_file": self.n_embeddings_per_file
            }, f)

    @classmethod
    def load(cls, storage_path: str):
        with open(storage_path + "/key2idx.json", "r") as f:
            key2idx = json.load(f)

        with open(storage_path + "/metadata.json", "r") as f:
            data = json.load(f)

        _cls = cls(
            storage_path=storage_path, 
            emb_dim=data["emb_dim"], 
            emb_dtype=cls.str2dtype[data["emb_dtype"]], 
            n_cached_values=data["n_cached_values"], 
            n_embeddings_per_file=data["n_embeddings_per_file"]
        )
        _cls.key2idx = key2idx

        return _cls

    def set(self, key: str, value: bytes):
        self.key2idx[key] = self.key2idx.get(key, len(self.key2idx))

        pos = self._get_pos_in_file(key=key)
        path = self.key2path(key)

        if (pos == 0) and (not os.path.exists(path)):
            # create new file
            with open(path, "wb") as f:
                f.write(value)
        else:
            # update current file
            # TODO: find smart way to fix cache on key update
            self.clear_cache()
            
            with open(path, "r+b") as f:
                f.seek(self.emb_size * pos)
                f.write(value)
    
    def get_embedding(self, key: str) -> np.ndarray | None:
        data = self.get(key=key)
        if data is None:
            return data
        
        return self.serializer.deserialize(data)
    
    def get_embeddings(self, keys: list[str]) -> tuple[Sequence[str], np.ndarray] | None:
        paths = [self.key2path(key) for key in keys]
        positions_in_files = [self._get_pos_in_file(key) for key in keys]

        triplets = sorted(zip(paths, positions_in_files, keys), key=lambda x: x[0])
        grouped_triplets = groupby(triplets, lambda x: x[0])

        result_keys = []
        result_embeddings = []
        for path, grouped_pair in grouped_triplets:
            pairs_in_file = [(pos, key) for _, pos, key in grouped_pair]

            try:
                with open(path, "rb") as f:
                    for pos, key in pairs_in_file:
                        f.seek(self.emb_size * pos)
                        result_embeddings.append(self.serializer.deserialize(f.read(self.emb_size)))
                        result_keys.append(key)
            except FileNotFoundError as e:
                continue
        
        if len(result_keys) == 0:
            return None
        
        return result_keys, np.stack(result_embeddings)

    def set_embedding(self, key: str, embedding: np.ndarray):
        bytes_array = self.serializer.serialize(embedding=embedding)
        self.set(key, bytes_array)
