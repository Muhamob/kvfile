import logging
import time
import pytest

import numpy as np
from tqdm import tqdm

from kvfile import EmbeddingsFileBulk


logger = logging.getLogger(__name__)


def test_dump_load(tmpdir):
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
        assert ((a - a_hat)**2).sum()**0.5 < 1e-5

    kvfile.dump()

    del kvfile

    kvfile = EmbeddingsFileBulk.load(tmpdir)

    for i in range(n_embeddings):
        a_hat = kvfile.get_embedding(str(i))
        a = embeddings[i]

        assert a_hat is not None
        assert ((a - a_hat)**2).sum()**0.5 < 1e-5
