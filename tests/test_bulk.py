import logging
import pytest

import numpy as np
from tqdm import tqdm

from kvfile import EmbeddingsFileBulk


logger = logging.getLogger(__name__)


def test_smoke(tmpdir):
    emb_dim = 128
    n_embeddings = 10
    dtype = np.float32

    random_gen = np.random.default_rng(42)
    embeddings = random_gen.random(size=(n_embeddings, emb_dim)).astype(dtype)

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_embeddings_per_file=4
    )

    for i in range(n_embeddings):
        kvfile.set_embedding(str(i), embedding=embeddings[i])
        print(kvfile.key2idx)
    
    for i in range(n_embeddings):
        a_hat = kvfile.get_embedding(str(i))
        a = embeddings[i]

        assert a_hat is not None
        logger.info(f"Diff in between vectors: {np.linalg.norm(a - a_hat)}")
        assert ((a - a_hat)**2).sum()**0.5 < 1e-5


@pytest.mark.parametrize("n_embedding", [10000])
@pytest.mark.parametrize("emb_dim", [128, 1024])
@pytest.mark.parametrize("n_embeddings_per_file", [4, 128])
def test_set(tmpdir, n_embedding, emb_dim, n_embeddings_per_file):
    dtype = np.float32

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_embeddings_per_file=n_embeddings_per_file
    )
    random_gen = np.random.default_rng(42)
    vectors = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)


@pytest.fixture()
def kvfile_4emb_per_file(tmpdir):
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_embeddings_per_file = 4

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_embeddings_per_file=n_embeddings_per_file
    )
    random_gen = np.random.default_rng(42)
    vectors = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)

    return kvfile


@pytest.fixture()
def kvfile_4emb_per_file_cached(tmpdir):
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_embeddings_per_file = 4
    n_cache = 128

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_cached_values=n_cache,
        n_embeddings_per_file=n_embeddings_per_file
    )
    random_gen = np.random.default_rng(42)
    vectors = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)

    return kvfile


@pytest.fixture()
def kvfile_1024emb_per_file(tmpdir):
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_embeddings_per_file = 1024

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_embeddings_per_file=n_embeddings_per_file
    )
    random_gen = np.random.default_rng(42)
    vectors = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)

    return kvfile


@pytest.fixture()
def kvfile_1024emb_per_file_cached(tmpdir):
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_embeddings_per_file = 1024
    n_cache = 128

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_cached_values=n_cache,
        n_embeddings_per_file=n_embeddings_per_file
    )
    random_gen = np.random.default_rng(42)
    vectors = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(vectors)):
        kvfile.set_embedding(str(i), e)

    return kvfile


@pytest.mark.parametrize("n_reads", [100000])
@pytest.mark.parametrize("kvfile", [
    "kvfile_4emb_per_file", 
    "kvfile_1024emb_per_file",
    "kvfile_4emb_per_file_cached", 
    "kvfile_1024emb_per_file_cached",
])
def test_get_random(n_reads, kvfile, request):
    kvfile = request.getfixturevalue(kvfile)
    random_gen = np.random.default_rng(42)

    logger.info("read values")
    for _ in tqdm(range(n_reads)):
        idx = random_gen.integers(10_000)
        kvfile.get(str(idx))
