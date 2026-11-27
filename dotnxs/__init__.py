"""DOTNXS single-file archive and conversion tools."""

from .archive import (
    DEFAULT_CHUNK_BYTES,
    MAGIC,
    NXS_VERSION,
    NxsArchive,
    NxsFormatError,
    NxsWriter,
)
from .parquet_dataset import NxsParquetDataset, NxsParquetEntryReader

__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "MAGIC",
    "NXS_VERSION",
    "NxsArchive",
    "NxsFormatError",
    "NxsParquetDataset",
    "NxsParquetEntryReader",
    "NxsWriter",
]

__version__ = "0.2.0.dev0"
