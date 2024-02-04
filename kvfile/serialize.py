from abc import ABC, abstractmethod
import io
import struct

import numpy as np


all_serializers = ["np.save","np.tobytes", "struct"]


class EmbeddingSerializer(ABC):
    @abstractmethod
    def serialize(self, embedding: np.ndarray) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, bytes_buffer: bytes) -> np.ndarray:
        pass


class EmbeddingSerializerFactory:
    def __init__(self):
        self.emb_dim = None
        self.dtype = None

    def set_dim(self, emb_dim: int) -> 'EmbeddingSerializerFactory':
        self.emb_dim = emb_dim
        return self
    
    def set_dtype(self, dtype: type) -> 'EmbeddingSerializerFactory':
        self.dtype = dtype
        return self
    
    def make_serializer(self, serializer: str) -> EmbeddingSerializer:
        if serializer == "np.save":
            return NumpySaveEmbeddingsSerializer()
        elif serializer == "np.tobytes":
            assert self.dtype is not None
            return NumpyToBytesEmbeddingsSerializer(dtype=self.dtype)
        elif serializer == "struct":
            assert self.dtype is not None
            assert self.emb_dim is not None
            return StructEmbeddingSerializer(dim=self.emb_dim, dtype=self.dtype)
        else:
            raise TypeError(f"{serializer} serializer not found")


class NumpySaveEmbeddingsSerializer(EmbeddingSerializer):
    def serialize(self, embedding: np.ndarray) -> bytes:
        assert embedding.ndim == 1

        buffer = io.BytesIO()
        np.save(buffer, embedding)
        buffer.seek(0)

        return buffer.read()
    
    def deserialize(self, bytes_buffer: bytes) -> np.ndarray:
        return np.load(io.BytesIO(bytes_buffer))
    

class NumpyToBytesEmbeddingsSerializer(EmbeddingSerializer):
    def __init__(self, dtype):
        self.dtype = dtype

    def serialize(self, embedding: np.ndarray) -> bytes:
        assert embedding.ndim == 1

        return embedding.tobytes()
    
    def deserialize(self, bytes_buffer: bytes) -> np.ndarray:
        return np.frombuffer(bytes_buffer, self.dtype)


class StructEmbeddingSerializer(EmbeddingSerializer):
    np2struct_dtype = {
        # floats
        np.float16: "e",
        np.float32: "f",
        np.float64: "d",

        # ints
        np.int16: "h",
        np.int32: "i",
        np.int64: "q",
    }

    def __init__(self, dim: int, dtype):
        self.endian = ">"
        self.dim = dim
        self.dtype = dtype

        self.fmt = f"{self.endian}" + self.np2struct_dtype[dtype] * self.dim

    def serialize(self, embedding: np.ndarray) -> bytes:
        assert embedding.ndim == 1

        return struct.pack(self.fmt, *embedding)

    def deserialize(self, bytes_array: bytes) -> np.ndarray:
        return np.array(struct.unpack(self.fmt, bytes_array), dtype=self.dtype)


