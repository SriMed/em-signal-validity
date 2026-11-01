"""Export a run artifact through notebook output when browser downloads fail.

A manifest records the complete file hash; each bounded chunk carries its own
hash and byte offset. Decode locally and verify the final manifest before use.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path


DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


def file_manifest(path, chunk_bytes=DEFAULT_CHUNK_BYTES):
    if chunk_bytes <= 0:
        raise ValueError('Chunk size must be positive')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    size = path.stat().st_size
    return dict(name=path.name, size=size, sha256=digest.hexdigest(),
                chunk_bytes=chunk_bytes, chunks=(size + chunk_bytes - 1) // chunk_bytes)


def file_chunk(path, index, chunk_bytes=DEFAULT_CHUNK_BYTES):
    if chunk_bytes <= 0 or index < 0:
        raise ValueError('Chunk size must be positive and index nonnegative')
    size = path.stat().st_size
    offset = index * chunk_bytes
    if offset >= size:
        raise ValueError('Chunk index is outside the artifact')
    with path.open('rb') as stream:
        stream.seek(offset)
        data = stream.read(chunk_bytes)
    return dict(name=path.name, index=index, offset=offset, size=len(data),
                total_size=size, sha256=hashlib.sha256(data).hexdigest(),
                data=base64.b64encode(data).decode('ascii'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--index', type=int, help='Omit for the complete-file manifest')
    parser.add_argument('--chunk-bytes', type=int, default=DEFAULT_CHUNK_BYTES)
    args = parser.parse_args()
    result = (file_manifest(args.path, args.chunk_bytes) if args.index is None
              else file_chunk(args.path, args.index, args.chunk_bytes))
    print('PHASE2_EXPORT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
