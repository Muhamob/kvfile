import io

import numpy as np


class NPEmbeddingsSerializer:
    @staticmethod
    def serialize(embeddings: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, embeddings)
        buffer.seek(0)

        return buffer.read()
    
    @staticmethod
    def deserialize(bytes_buffer: bytes) -> np.ndarray:
        return np.load(io.BytesIO(bytes_buffer))