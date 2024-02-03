# KVFile - Key Value on file storage
- Key value хранилище с хранением данных на диске и кэшем для частотных ключей

# Example
```python
import numpy as np
from kvfile import EmbeddingsFile


emb_dim = 128
dtype = np.float32

a = np.random.rand(emb_dim).astype(dtype)

kvfile = EmbeddingsFile(
    storage_path=tmpdir, 
    emb_dim=emb_dim, 
    emb_dtype=dtype,
    n_cached_values=1000, 
    serializer="struct"  # or "numpy"
)
kvfile.set_embedding("1", a)
a_hat = kvfile.get_embedding("1")
```