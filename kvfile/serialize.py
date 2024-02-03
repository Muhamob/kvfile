from abc import ABC, abstractmethod
import io
import struct

import numpy as np


class EmbeddingSerializer(ABC):
    @abstractmethod
    def serialize(self, embedding: np.ndarray) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, bytes_buffer: bytes) -> np.ndarray:
        pass


class NumpyEmbeddingsSerializer(EmbeddingSerializer):
    def serialize(self, embedding: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, embedding)
        buffer.seek(0)

        return buffer.read()
    
    def deserialize(self, bytes_buffer: bytes) -> np.ndarray:
        return np.load(io.BytesIO(bytes_buffer))
    

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


