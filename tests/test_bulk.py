import logging
import time
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


@pytest.mark.parametrize("n_embedding", [10_000])
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


@pytest.mark.parametrize("n_reads", [100_000])
@pytest.mark.parametrize("kvfile", [
    "kvfile_4emb_per_file", 
    "kvfile_1024emb_per_file",
    "kvfile_4emb_per_file_cached", 
    "kvfile_1024emb_per_file_cached",
])
def test_get_embedding_random(n_reads, kvfile, request):
    kvfile = request.getfixturevalue(kvfile)
    random_gen = np.random.default_rng(42)

    logger.info("read values")
    for _ in tqdm(range(n_reads)):
        idx = random_gen.integers(10_000)
        kvfile.get(str(idx))


@pytest.mark.parametrize("n_reads", [100_000])
@pytest.mark.parametrize("kvfile", [
    "kvfile_4emb_per_file", 
    "kvfile_1024emb_per_file",
    "kvfile_4emb_per_file_cached", 
    "kvfile_1024emb_per_file_cached",
])
def test_get_embeddings_random(n_reads, kvfile, request):
    kvfile = request.getfixturevalue(kvfile)
    random_gen = np.random.default_rng(42)

    logger.info("read values")
    ids = [str(random_gen.integers(10_000)) for _ in range(n_reads)]
    
    keys, embeddings = kvfile.get_embeddings(ids)

    print(embeddings)

    assert len(keys) == n_reads
    assert embeddings.shape[0] == n_reads
    assert embeddings.shape[1] == kvfile.emb_dim


@pytest.mark.parametrize("n_cache", [None, 128, 1024])
def test_get_embeddings_eq(tmpdir, n_cache):
    # setup
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_embeddings_per_file = 1024
    n_reads = 1_000_000

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}, n_cache={n_cache}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_cached_values=n_cache,
        n_embeddings_per_file=n_embeddings_per_file
    )
    random_gen = np.random.default_rng(42)
    input_embeddings = random_gen.random(size=(n_embedding, emb_dim)).astype(dtype)

    for i, e in tqdm(enumerate(input_embeddings)):
        kvfile.set_embedding(str(i), e)
    
    # test getting 100_000 random embeddings
    ids = [str(random_gen.integers(n_embedding)) for _ in range(n_reads)]
    start_dt = time.time()
    result = kvfile.get_embeddings(ids)
    time_diff = time.time() - start_dt
    logger.info(f"Time spent getting embeddings: {time_diff} = {n_reads / time_diff} reads per sec.")
    assert result is not None

    keys, read_embeddings = result

    keys = np.array([int(key) for key in keys])

    # check equal
    assert ((input_embeddings[keys] - read_embeddings)**2).sum() ** 0.5 < 1e-5


@pytest.mark.parametrize("n_cache", [None, 128, 1024])
def test_get_embedding_eq(tmpdir, n_cache):
    # setup
    dtype = np.float32
    n_embedding = 10_000
    emb_dim = 128
    n_embeddings_per_file = 1024
    n_reads = 1_000_000

    logger.info(f"n_embedding={n_embedding}, dim={emb_dim}, n_cache={n_cache}")

    kvfile = EmbeddingsFileBulk(
        storage_path=tmpdir, 
        emb_dim=emb_dim, 
        emb_dtype=dtype,
        n_cached_values=n_cache,
        n_embeddings_per_file=n_embeddings_per_file
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
