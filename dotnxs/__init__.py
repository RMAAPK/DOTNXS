"""DOTNXS single-file archive and conversion tools."""

from .archive import (
    DEFAULT_CHUNK_BYTES,
    MAGIC,
    NXS_VERSION,
    NxsArchive,
    NxsFormatError,
    NxsWriter,
)

__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "MAGIC",
    "NXS_VERSION",
    "NxsArchive",
    "NxsFormatError",
    "NxsWriter",
]

__version__ = "0.1.0"
