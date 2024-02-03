# KVFile - Key Value on file storage
- Key value хранилище с хранением данных на диске и кэшем для частотных ключей

# Example
```python
kvfile = KVFile(cache_size: int)
kvfile.set("key", binary)
binary = kvfile.get("key")
```