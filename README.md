# KVFile - Key Value Storage on File
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


# Benchmark

On mac m1 pro 14 inch 16Gb

## Serialize
```python
def serialize_io(a):
    # 7.29 µs ± 47.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    # 2176
    buf = io.BytesIO()
    np.save(buf, a)
    buf.seek(0)
    return buf.read()


def serialize_tobytes(a):
    # 161 ns ± 0.448 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    # 512 * 4 = 2048
    return a.tobytes()


fmt = ">"+"f"*512
def serialize_struct(a):
    # 23.4 µs ± 56.9 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    # size 512 * 4 = 2048
    return struct.pack(fmt, *a)


fmt = ">"+"f"*512
```

## Serialize + Deserialize
```python
def deserialize_struct(a):
    # 44.5 µs ± 70.7 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    return np.array(struct.unpack(fmt, struct.pack(fmt, *a)))


def deserialize_tobytes(a):
    # 416 ns ± 0.53 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    return np.frombuffer(a.tobytes(), np.float32)


def deserialize_io(a):
    # 37.1 µs ± 110 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    buf = io.BytesIO()
    np.save(buf, a)
    buf.seek(0)
    a_bytes = buf.read()
    buf = io.BytesIO(a_bytes)
    return np.load(buf)
```