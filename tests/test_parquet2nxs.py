import importlib.util
import tempfile
import unittest
from pathlib import Path

from dotnxs.archive import NxsArchive
from dotnxs.parquet2nxs import convert_parquet
from dotnxs.parquet_dataset import NxsParquetDataset


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "install dotnxs[parquet] to test Parquet packing")
class ParquetToNxsTests(unittest.TestCase):
    def test_packs_partitioned_parquet_shards_and_row_group_summaries(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "rules"
            shard_dir = root / "topic=networking"
            shard_dir.mkdir(parents=True)
            shard = shard_dir / "part-000.parquet"
            table = pa.table(
                {
                    "rule_id": ["r1", "r2", "r3"],
                    "question": ["q1", "q2", "q3"],
                    "answer": ["a1", "a2", "a3"],
                }
            )
            pq.write_table(table, shard, row_group_size=2)
            output = Path(temp) / "rules.nxs"

            result = convert_parquet(root, output, chunk_bytes=128)

            self.assertEqual(result["parquet_files"], 1)
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["row_group_count"], 2)
            with NxsArchive(output) as archive:
                entry_name = "dataset/topic=networking/part-000.parquet"
                entry = archive.get_entry(entry_name)
                self.assertEqual(entry["kind"], "parquet_shard")
                self.assertEqual(entry["metadata"]["format"], "parquet")
                self.assertEqual(entry["metadata"]["row_count"], 3)
                self.assertEqual(entry["metadata"]["row_group_count"], 2)
                self.assertEqual([g["rows"] for g in entry["metadata"]["row_groups"]], [2, 1])
                self.assertEqual(b"".join(archive.iter_entry(entry_name)), shard.read_bytes())
                dataset = NxsParquetDataset(archive)
                self.assertEqual(dataset.list_shards()[0]["row_group_count"], 2)
                second_group = dataset.read_row_group(entry_name, 1)
                self.assertEqual(second_group.column("rule_id").to_pylist(), ["r3"])

    def test_accepts_a_single_parquet_shard_and_rejects_bad_magic(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root / "single.parquet"
            pq.write_table(pa.table({"rule": ["keep"]}), shard)
            output = root / "single.nxs"
            result = convert_parquet(shard, output)
            self.assertEqual(result["parquet_files"], 1)

            invalid = root / "not-parquet.parquet"
            invalid.write_bytes(b"not parquet data")
            with self.assertRaisesRegex(ValueError, "PAR1"):
                convert_parquet(invalid, root / "invalid.nxs")


if __name__ == "__main__":
    unittest.main()
