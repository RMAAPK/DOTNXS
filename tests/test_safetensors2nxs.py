import json
import struct
import tempfile
import unittest
from pathlib import Path

from dotnxs.archive import HEADER_STRUCT, MAGIC, NxsArchive, NxsFormatError
from dotnxs.safetensors2nxs import convert_file


class SafetensorsToNxsTests(unittest.TestCase):
    def _make_safetensors(self, path: Path) -> dict[str, bytes]:
        weight = struct.pack("<ff", 1.25, -2.0)
        ids = bytes(range(8))
        header = {
            "__metadata__": {"format": "unit-test"},
            "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, len(weight)]},
            "token_ids": {
                "dtype": "U8",
                "shape": [len(ids)],
                "data_offsets": [len(weight), len(weight) + len(ids)],
            },
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)
        with open(path, "wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            output.write(weight)
            output.write(ids)
        return {"weight": weight, "token_ids": ids}

    def test_conversion_is_single_archive_and_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.safetensors"
            target = Path(directory) / "tiny.nxs"
            tensors = self._make_safetensors(source)

            result = convert_file(source, target, chunk_bytes=3)
            self.assertEqual(result["tensor_count"], 2)
            self.assertTrue(target.is_file())
            self.assertFalse(Path(str(target) + ".part").exists())

            with open(target, "rb") as file:
                header = HEADER_STRUCT.unpack(file.read(HEADER_STRUCT.size))
            self.assertEqual(header[0], MAGIC)
            self.assertEqual((header[1], header[2]), (0, 2))

            with NxsArchive(target) as archive:
                self.assertEqual(archive.manifest["source"]["format"], "safetensors")
                self.assertEqual(set(item["name"] for item in archive.list_entries()), set(tensors))
                archive.verify()
                for name, expected in tensors.items():
                    actual = b"".join(archive.iter_entry(name))
                    self.assertEqual(actual, expected)
                weight_entry = archive.get_entry("weight")
                self.assertEqual(weight_entry["dtype"], "F32")
                self.assertEqual(weight_entry["shape"], [2])
                self.assertEqual(len(weight_entry["chunks"]), 3)

    def test_rejects_corrupt_archive_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.safetensors"
            target = Path(directory) / "tiny.nxs"
            self._make_safetensors(source)
            convert_file(source, target, chunk_bytes=4)
            with open(target, "r+b") as file:
                file.seek(0)
                file.write(b"NOTNXS")
            with self.assertRaises(NxsFormatError):
                NxsArchive(target)

    def test_refuses_existing_output_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.safetensors"
            target = Path(directory) / "tiny.nxs"
            self._make_safetensors(source)
            convert_file(source, target, chunk_bytes=4)
            with self.assertRaises(FileExistsError):
                convert_file(source, target, chunk_bytes=4)


if __name__ == "__main__":
    unittest.main()
