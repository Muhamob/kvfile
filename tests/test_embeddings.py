import logging
import time
import pytest

import numpy as np
from tqdm import tqdm

from kvfile import EmbeddingsFile
from kvfile.serialize import StructEmbeddingSerializer, all_serializers


logger = logging.getLogger(__name__)


@pytest.fixture()
def serializer():
    return StructEmbeddingSerializer(dim=128, dtype=np.float32)


@pytest.mark.parametrize("serializer", all_serializers)
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


@pytest.mark.parametrize("n_embedding", [10000])
@pytest.mark.parametrize("dim", [128, 1536])
@pytest.mark.parametrize("serializer", all_serializers)
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


@pytest.mark.parametrize("n_embedding", [10000])
@pytest.mark.parametrize("dim", [128, 1536])
@pytest.mark.parametrize("serializer", all_serializers)
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


@pytest.mark.parametrize("n_cache", [None, 128, 1024])
def test_get_embedding_eq(tmpdir, n_cache):
    # setup
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_reads = 1_000_000

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}, n_cache={n_cache}")

    kvfile = EmbeddingsFile(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_cached_values=n_cache,
    )
    random_gen = np.random.default_rng(42)
    input_embeddings = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(input_embeddings)):
        kvfile.set_embedding(str(i), e)
    
    # test getting 100_000 random embeddings
    keys = [str(random_gen.integers(n_embedding)) for _ in range(n_reads)]
    
    start_dt = time.time()
    read_embeddings = []
    for key in keys:
        read_embeddings.append(kvfile.get_embedding(key))
    read_embeddings = np.stack(read_embeddings)

    time_diff = time.time() - start_dt
    logger.info(f"Time spent getting embeddings: {time_diff} = {n_reads / time_diff} reads per sec.")

    ids = np.array([int(key) for key in keys])

    # check equal
    assert ((input_embeddings[ids] - read_embeddings)**2).sum() ** 0.5 < 1e-5


@pytest.mark.parametrize("n_cached_values", [None, 128])
def test_set_update(tmpdir, n_cached_values):
    dim = 128
    dtype = np.float32

    kvfile = EmbeddingsFile(
        storage_path=tmpdir, 
        emb_dim=dim, 
        emb_dtype=dtype,
        n_cached_values=n_cached_values, 
    )
    vectors = np.random.rand(2, dim).astype(dtype)

    kvfile.set_embedding("0", vectors[0])
    kvfile.set("1", vectors[1])
    # activate cache
    kvfile.get("1")
    # replace value
    updated_vector = vectors[1] * 1000
    kvfile.set("1", updated_vector)

    assert np.abs(updated_vector - kvfile.get_embedding("1")).sum() < 1e-5