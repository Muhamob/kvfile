import io
import logging
import pytest

import numpy as np
from tqdm import tqdm

from kvfile.kvfile import EmbeddingsFile
from kvfile.serialize import StructEmbeddingSerializer


logger = logging.getLogger(__name__)
serializers = ["struct", "numpy", "numpybuffer"]


@pytest.fixture()
def serializer():
    return StructEmbeddingSerializer(dim=128, dtype=np.float32)


@pytest.mark.parametrize("serializer", serializers)
def test_smoke(tmpdir, serializer):
    emb_dim = 128
    dtype = np.float32

    a = np.random.rand(emb_dim).astype(dtype)

    kvfile = EmbeddingsFile(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_cached_values=1000, 
        serializer=serializer
    )
    kvfile.set_embedding("1", a)
    a_hat = kvfile.get_embedding("1")

    assert a_hat is not None
    logger.info(f"Diff in between vectors: {np.linalg.norm(a - a_hat)}")
    assert ((a - a_hat)**2).sum()**0.5 < 1e-5


@pytest.mark.parametrize("n_embedding", [100000])
@pytest.mark.parametrize("dim", [128, 1536])
@pytest.mark.parametrize("serializer", serializers)
def test_set(tmpdir, n_embedding, dim, serializer):
    logger.info(f"n_embedding={n_embedding}, dim={dim}, serializer={serializer}")
    kvfile = EmbeddingsFile(
        storage_path=tmpdir, 
        emb_dim=dim, 
        emb_dtype=np.float32,
        n_cached_values=1000, 
        serializer=serializer
    )
    vectors = np.random.rand(n_embedding, dim)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)


@pytest.mark.parametrize("n_embedding", [100000])
@pytest.mark.parametrize("dim", [128, 1536])
@pytest.mark.parametrize("serializer", serializers)
def test_get(tmpdir, n_embedding, dim, serializer):
    logger.info(f"n_embedding={n_embedding}, dim={dim}, serializer={serializer}")
    kvfile = EmbeddingsFile(
        storage_path=tmpdir, 
        emb_dim=dim, 
        emb_dtype=np.float32,
        n_cached_values=1000, 
        serializer=serializer
    )
    vectors = np.random.rand(n_embedding, dim)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.get_embedding(str(i))