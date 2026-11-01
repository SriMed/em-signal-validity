"""Verify artifact reconstruction independently of browser download behavior."""
import base64
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('export_artifact', Path(__file__).parents[1] / 'scripts/export_artifact.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ArtifactExportTests(unittest.TestCase):
    def test_chunks_reconstruct_original_file_and_manifest_hash(self):
        original = bytes(range(256)) * 513 + b'partial final chunk'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact.bin'
            path.write_bytes(original)
            manifest = export.file_manifest(path, chunk_bytes=4096)
            restored = bytearray()
            for i in range(manifest['chunks']):
                chunk = export.file_chunk(path, i, chunk_bytes=4096)
                data = base64.b64decode(chunk['data'], validate=True)
                self.assertEqual(chunk['offset'], len(restored))
                self.assertEqual(chunk['size'], len(data))
                self.assertEqual(chunk['sha256'], hashlib.sha256(data).hexdigest())
                restored.extend(data)
            self.assertEqual(bytes(restored), original)
            self.assertEqual(manifest['size'], len(restored))
            self.assertEqual(manifest['sha256'], hashlib.sha256(restored).hexdigest())

    def test_reject_out_of_range_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact.bin'
            path.write_bytes(b'abc')
            for index in [-1, 1]:
                with self.assertRaises(ValueError):
                    export.file_chunk(path, index, chunk_bytes=3)
            with self.assertRaises(ValueError):
                export.file_manifest(path, chunk_bytes=0)
