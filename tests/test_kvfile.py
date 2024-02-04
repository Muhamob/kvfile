import logging
import io

import pytest
import numpy as np
from tqdm import tqdm

from kvfile import KVFile


logger = logging.getLogger(__name__)


@pytest.fixture()
def kvfile(tmpdir):
    return KVFile(tmpdir, 1000)


@pytest.fixture()
def kvfile1000_load1000(tmpdir):
    kvfile = KVFile(tmpdir, 1000)
    vectors = np.random.rand(1000, 128)
    
    for i, e in enumerate(vectors):
        buf = io.BytesIO()
        np.save(buf, e)
        buf.seek(0)

        kvfile.set(str(i), buf.read())

    return kvfile


@pytest.fixture()
def kvfile100_load1000(tmpdir):
    kvfile = KVFile(tmpdir, 100)
    vectors = np.random.rand(1000, 128)
    
    for i, e in enumerate(vectors):
        buf = io.BytesIO()
        np.save(buf, e)
        buf.seek(0)

        kvfile.set(str(i), buf.read())

    return kvfile


def test_smoke(kvfile: KVFile):
    assert kvfile.get("this_key_doesnt_exists") is None

    kvfile.set("1", b'000')
    assert isinstance(kvfile.get("1"), bytes)
    assert kvfile.get("1") == b'000'


@pytest.mark.parametrize("n_embedding", [10000])
@pytest.mark.parametrize("dim", [128, 1536])
def test_set(tmpdir, n_embedding, dim):
    kvfile = KVFile(tmpdir, None)
    vectors = np.random.rand(n_embedding, dim)

    print(n_embedding, dim)

    for i, e in tqdm(enumerate(vectors)):
        buf = io.BytesIO()
        np.save(buf, e)
        buf.seek(0)

        kvfile.set(str(i), buf.read())


@pytest.mark.parametrize("n_reads", [100000])
@pytest.mark.parametrize("n_embedding", [10000])
@pytest.mark.parametrize("n_cache", [100, 10000])
def test_get(n_reads, n_embedding, n_cache, tmpdir):
    logger.info(f"n_reads={n_reads}, n_embedding={n_embedding}, n_cache={n_cache}")
    kvfile = KVFile(tmpdir, n_cache)
    vectors = np.random.rand(n_embedding, 128)
    
    logger.info("Set values")
    for i, e in tqdm(enumerate(vectors)):
        buf = io.BytesIO()
        np.save(buf, e)
        buf.seek(0)

        kvfile.set(str(i), buf.read())

    idx = np.random.randint(0, n_embedding, n_reads)

    logger.info("read values")
    for id_ in tqdm(idx):
        kvfile.get(id_)
