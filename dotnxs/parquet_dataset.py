"""Range-backed Parquet row-group access for Parquet shards inside an NXS archive."""

from __future__ import annotations

import io
from typing import Any


class NxsParquetEntryReader(io.RawIOBase):
    """Seekable file-like view of one Parquet shard stored as an NXS entry."""

    def __init__(self, archive: Any, entry_name: str) -> None:
        self.archive = archive
        self.entry_name = entry_name
        entry = archive.get_entry(entry_name)
        metadata = entry.get("metadata", {})
        if entry.get("kind") != "parquet_shard" or metadata.get("format") != "parquet":
            raise ValueError(f"NXS entry {entry_name!r} is not a Parquet shard")
        self._size = entry["nbytes"]
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"invalid seek mode: {whence}")
        if not isinstance(position, int) or position < 0:
            raise ValueError("Parquet shard seek position must be a non-negative integer")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed Parquet shard reader")
        remaining = max(0, self._size - self._position)
        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        if size == 0:
            return b""
        data = self.archive.read_entry_range(self.entry_name, self._position, size)
        self._position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class NxsParquetDataset:
    """Read Parquet shards/row groups without extracting complete files first."""

    def __init__(self, archive: Any) -> None:
        self.archive = archive
        self.shards = {
            entry["name"]: entry
            for entry in archive.list_entries()
            if entry.get("kind") == "parquet_shard"
            and entry.get("metadata", {}).get("format") == "parquet"
        }

    def list_shards(self) -> list[dict[str, Any]]:
        """Return small manifest summaries for available Parquet shards."""
        return [
            {
                "name": name,
                "source_path": entry.get("metadata", {}).get("source_path"),
                "nbytes": entry["nbytes"],
                "row_count": entry.get("metadata", {}).get("row_count"),
                "row_group_count": entry.get("metadata", {}).get("row_group_count"),
                "schema": entry.get("metadata", {}).get("schema"),
            }
            for name, entry in self.shards.items()
        ]

    def read_row_group(
        self,
        shard_name: str,
        row_group: int,
        *,
        columns: list[str] | None = None,
    ):
        """Decode one Parquet row group, fetching only its intersecting NXS chunks."""
        if shard_name not in self.shards:
            raise KeyError(f"Parquet shard not found in NXS archive: {shard_name}")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet reading requires `pip install 'dotnxs[parquet]'`") from exc

        reader = NxsParquetEntryReader(self.archive, shard_name)
        source = pa.PythonFile(reader)
        try:
            parquet = pq.ParquetFile(source)
            if not isinstance(row_group, int) or isinstance(row_group, bool):
                raise TypeError("row_group must be an integer")
            if not 0 <= row_group < parquet.metadata.num_row_groups:
                raise IndexError(f"row_group index {row_group} is outside the Parquet shard")
            return parquet.read_row_group(row_group, columns=columns)
        finally:
            source.close()


__all__ = ["NxsParquetDataset", "NxsParquetEntryReader"]
