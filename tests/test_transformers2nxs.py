import json
import struct
import tempfile
import unittest
from pathlib import Path

from dotnxs.archive import NxsArchive
from dotnxs.transformers2nxs import convert_directory


def _write_safetensors(path: Path, tensor_name: str, payload: bytes) -> None:
    header = {
        tensor_name: {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        output.write(payload)


class TransformersToNxsTests(unittest.TestCase):
    def test_packs_shards_and_assets_into_one_lossless_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            output = Path(directory) / "model.nxs"
            base_payload = bytes(range(8))
            mtp_payload = b"mtp-data"
            _write_safetensors(root / "model.safetensors", "weight", base_payload)
            _write_safetensors(root / "mtp.safetensors", "weight", mtp_payload)
            (root / "config.json").write_text(
                json.dumps({"model_type": "fixture", "architectures": ["FixtureModel"]}),
                encoding="utf-8",
            )
            (root / "tokenizer.json").write_text('{"fixture":true}', encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model.safetensors"}}),
                encoding="utf-8",
            )

            result = convert_directory(root, output, chunk_bytes=3)
            self.assertEqual(result["weight_files"], 2)
            self.assertEqual(result["asset_files"], 3)
            self.assertEqual(result["tensor_count"], 2)

            with NxsArchive(output) as archive:
                self.assertEqual(archive.manifest["source"]["format"], "transformers")
                self.assertEqual(archive.manifest["source"]["model_type"], "fixture")
                archive.verify()
                self.assertEqual(b"".join(archive.iter_entry("weight")), base_payload)
                self.assertEqual(b"".join(archive.iter_entry("mtp/weight")), mtp_payload)
                self.assertEqual(
                    b"".join(archive.iter_entry("asset/tokenizer.json")),
                    b'{"fixture":true}',
                )

    def test_rejects_index_referencing_missing_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            _write_safetensors(root / "model.safetensors", "weight", b"1234")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"missing": "model.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "absent from the shard header"):
                convert_directory(root, Path(directory) / "out.nxs")


if __name__ == "__main__":
    unittest.main()
