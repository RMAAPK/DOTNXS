"""DOTNXS v0.2 single-file archive writer and validator.

A .nxs file is one archive. Payload chunks are internal byte ranges indexed by
an authenticated JSON manifest; they are never separate .nxs files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

try:
    from ._core import NativeWriter as _NativeWriter
except ImportError:  # Source-tree tests can use the portable Python fallback.
    _NativeWriter = None

MAGIC = bytes.fromhex("4E 58 53 00 62 62 96 6D 2F 13 49 C9 86 91 C5 22 DC E3 08 43")
NXS_VERSION = (0, 2)
HEADER_STRUCT = struct.Struct("<20sHHIIQQ32s")
HEADER_SIZE = HEADER_STRUCT.size  # 80 bytes
ALIGNMENT = 4096
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
COPY_BUFFER_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NxsFormatError(ValueError):
    """Raised when an NXS archive is malformed or unsupported."""


def _align_file(file: BinaryIO, alignment: int = ALIGNMENT) -> None:
    padding = (-file.tell()) % alignment
    if padding:
        file.write(b"\0" * padding)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("entry name must be a non-empty string")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or any(piece == ".." for piece in normalized.split("/")):
        raise ValueError(f"unsafe entry name: {name!r}")


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class NxsWriter:
    """Create one seekable .nxs archive using bounded-memory streaming."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        source: dict | None = None,
        archive_metadata: dict | None = None,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be greater than zero")
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path = self.path.with_name(self.path.name + ".part")
        self._native_writer = (
            _NativeWriter(str(self.path), chunk_bytes) if _NativeWriter is not None else None
        )
        self._file = None
        if self._native_writer is None:
            self._file = open(self._tmp_path, "w+b")
            self._file.write(b"\0" * HEADER_SIZE)
        self._source = source or {}
        self._archive_metadata = archive_metadata or {}
        self._chunk_bytes = chunk_bytes
        self._entries: list[dict] = []
        self._names: set[str] = set()
        self._finished = False

    def add_stream(
        self,
        name: str,
        source: BinaryIO,
        size: int,
        *,
        kind: str = "asset",
        source_offset: int = 0,
        dtype: str | None = None,
        shape: list[int] | tuple[int, ...] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Copy a byte range into internal aligned chunks without buffering it all."""
        if self._finished:
            raise RuntimeError("archive writer is already finalized")
        _validate_name(name)
        if name in self._names:
            raise ValueError(f"duplicate NXS entry name: {name}")
        if not isinstance(kind, str) or not kind:
            raise ValueError("entry kind must be a non-empty string")
        if not _is_nonnegative_int(size) or not _is_nonnegative_int(source_offset):
            raise ValueError("size and source_offset must be non-negative integers")

        if self._native_writer is not None:
            source_path = getattr(source, "name", None)
            if not isinstance(source_path, (str, os.PathLike)):
                raise TypeError("native writer requires a path-backed source stream")
            entry = self._native_writer.add_file_range(
                name,
                os.fspath(source_path),
                source_offset,
                size,
                kind,
                dtype or "",
                list(shape or []),
                metadata or {},
            )
            self._entries.append(entry)
            self._names.add(name)
            return entry

        source.seek(source_offset)
        whole_hash = hashlib.sha256()
        chunks: list[dict] = []
        logical_offset = 0

        while logical_offset < size:
            chunk_length = min(self._chunk_bytes, size - logical_offset)
            _align_file(self._file)
            file_offset = self._file.tell()
            chunk_hash = hashlib.sha256()
            remaining = chunk_length
            while remaining:
                block = source.read(min(COPY_BUFFER_BYTES, remaining))
                if not block:
                    raise EOFError(
                        f"source ended early while reading {name!r}: "
                        f"needed {remaining} more bytes"
                    )
                self._file.write(block)
                chunk_hash.update(block)
                whole_hash.update(block)
                remaining -= len(block)
            chunks.append(
                {
                    "offset": file_offset,
                    "length": chunk_length,
                    "logical_offset": logical_offset,
                    "sha256": chunk_hash.hexdigest(),
                }
            )
            logical_offset += chunk_length

        entry = {
            "name": name,
            "kind": kind,
            "nbytes": size,
            "sha256": whole_hash.hexdigest(),
            "chunks": chunks,
        }
        if dtype is not None:
            entry["dtype"] = dtype
        if shape is not None:
            entry["shape"] = list(shape)
        if metadata:
            entry["metadata"] = metadata
        self._entries.append(entry)
        self._names.add(name)
        return entry

    def add_bytes(
        self,
        name: str,
        data: bytes,
        *,
        kind: str = "asset",
        dtype: str | None = None,
        shape: list[int] | tuple[int, ...] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if self._native_writer is not None:
            if self._finished:
                raise RuntimeError("archive writer is already finalized")
            _validate_name(name)
            if name in self._names:
                raise ValueError(f"duplicate NXS entry name: {name}")
            entry = self._native_writer.add_bytes(
                name,
                data,
                kind,
                dtype or "",
                list(shape or []),
                metadata or {},
            )
            self._entries.append(entry)
            self._names.add(name)
            return entry
        return self.add_stream(
            name,
            io.BytesIO(data),
            len(data),
            kind=kind,
            dtype=dtype,
            shape=shape,
            metadata=metadata,
        )

    def finalize(self) -> Path:
        if self._finished:
            return self.path
        try:
            manifest = {
                "format": "DOTNXS",
                "version": list(NXS_VERSION),
                "source": self._source,
                "archive_metadata": self._archive_metadata,
                "entries": self._entries,
            }
            manifest_bytes = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise ValueError("manifest exceeds the v0.2 size limit")

            if self._native_writer is not None:
                self._native_writer.finalize(manifest_bytes)
                self._finished = True
                return self.path

            _align_file(self._file)
            manifest_offset = self._file.tell()
            self._file.write(manifest_bytes)
            manifest_digest = hashlib.sha256(manifest_bytes).digest()

            self._file.seek(0)
            self._file.write(
                HEADER_STRUCT.pack(
                    MAGIC,
                    NXS_VERSION[0],
                    NXS_VERSION[1],
                    HEADER_SIZE,
                    0,  # flags reserved; must be zero in v0.2
                    manifest_offset,
                    len(manifest_bytes),
                    manifest_digest,
                )
            )
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            os.replace(self._tmp_path, self.path)
            self._finished = True
            return self.path
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if self._native_writer is not None:
            self._native_writer.abort()
            self._finished = True
            return
        if self._file is not None and not self._file.closed:
            self._file.close()
        try:
            self._tmp_path.unlink(missing_ok=True)
        finally:
            self._finished = True

    def __enter__(self) -> "NxsWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()
        return False


class NxsArchive:
    """Read and validate the NXS header/manifest; stream internal entries."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._file = open(self.path, "rb")
        self._file_size = self.path.stat().st_size
        try:
            raw_header = self._file.read(HEADER_SIZE)
            if len(raw_header) != HEADER_SIZE:
                raise NxsFormatError("truncated NXS header")
            (
                magic,
                major,
                minor,
                header_size,
                flags,
                manifest_offset,
                manifest_length,
                expected_digest,
            ) = HEADER_STRUCT.unpack(raw_header)
            if magic != MAGIC:
                raise NxsFormatError("not a DOTNXS archive: magic mismatch")
            if (major, minor) != NXS_VERSION:
                raise NxsFormatError(f"unsupported NXS version {major}.{minor}")
            if header_size != HEADER_SIZE:
                raise NxsFormatError(f"invalid header size {header_size}")
            if flags != 0:
                raise NxsFormatError("nonzero reserved header flags")
            if not _is_nonnegative_int(manifest_offset) or not _is_nonnegative_int(manifest_length):
                raise NxsFormatError("invalid manifest range")
            manifest_end = manifest_offset + manifest_length
            if (
                manifest_offset < HEADER_SIZE
                or manifest_length == 0
                or manifest_length > MAX_MANIFEST_BYTES
                or manifest_end != self._file_size
            ):
                raise NxsFormatError("manifest range is outside the archive or not at EOF")
            self._file.seek(manifest_offset)
            manifest_bytes = self._file.read(manifest_length)
            if len(manifest_bytes) != manifest_length:
                raise NxsFormatError("truncated NXS manifest")
            actual_digest = hashlib.sha256(manifest_bytes).digest()
            if actual_digest != expected_digest:
                raise NxsFormatError("manifest SHA-256 mismatch")
            try:
                self.manifest = json.loads(manifest_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NxsFormatError("invalid UTF-8 JSON manifest") from exc
            if self.manifest.get("format") != "DOTNXS":
                raise NxsFormatError("manifest format marker is missing")
            if self.manifest.get("version") != list(NXS_VERSION):
                raise NxsFormatError("manifest and header versions disagree")
            self.manifest_offset = manifest_offset
            self.entries = self.manifest.get("entries")
            self._entry_map = self._validate_entries()
        except Exception:
            self._file.close()
            raise

    def _validate_entries(self) -> dict[str, dict]:
        if not isinstance(self.entries, list):
            raise NxsFormatError("manifest entries must be a list")
        result: dict[str, dict] = {}
        ranges: list[tuple[int, int, str]] = []
        for entry in self.entries:
            if not isinstance(entry, dict):
                raise NxsFormatError("manifest entry must be an object")
            name = entry.get("name")
            try:
                _validate_name(name)
            except ValueError as exc:
                raise NxsFormatError(str(exc)) from exc
            if name in result:
                raise NxsFormatError(f"duplicate entry name {name!r}")
            size = entry.get("nbytes")
            if not _is_nonnegative_int(size):
                raise NxsFormatError(f"invalid byte length for {name!r}")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise NxsFormatError(f"invalid SHA-256 for {name!r}")
            chunks = entry.get("chunks")
            if not isinstance(chunks, list):
                raise NxsFormatError(f"invalid chunks list for {name!r}")
            logical_position = 0
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise NxsFormatError(f"invalid chunk record for {name!r}")
                offset = chunk.get("offset")
                length = chunk.get("length")
                logical_offset = chunk.get("logical_offset")
                chunk_digest = chunk.get("sha256")
                if not all(_is_nonnegative_int(v) for v in (offset, length, logical_offset)):
                    raise NxsFormatError(f"invalid chunk range for {name!r}")
                if length == 0 or logical_offset != logical_position:
                    raise NxsFormatError(f"non-contiguous logical chunks for {name!r}")
                if offset < HEADER_SIZE or offset % ALIGNMENT != 0 or offset + length > self.manifest_offset:
                    raise NxsFormatError(f"chunk outside payload area for {name!r}")
                if not isinstance(chunk_digest, str) or not _SHA256_RE.fullmatch(chunk_digest):
                    raise NxsFormatError(f"invalid chunk SHA-256 for {name!r}")
                logical_position += length
                ranges.append((offset, offset + length, name))
            if logical_position != size:
                raise NxsFormatError(f"chunk lengths do not match entry size for {name!r}")
            result[name] = entry
        ranges.sort()
        previous_end = HEADER_SIZE
        for start, end, name in ranges:
            if start < previous_end:
                raise NxsFormatError(f"overlapping payload chunks near entry {name!r}")
            previous_end = end
        return result

    def list_entries(self) -> list[dict]:
        return list(self.entries)

    def get_entry(self, name: str) -> dict:
        try:
            return self._entry_map[name]
        except KeyError as exc:
            raise KeyError(f"NXS entry not found: {name}") from exc

    def iter_entry(self, name: str, *, verify: bool = True) -> Iterator[bytes]:
        entry = self.get_entry(name)
        whole_hash = hashlib.sha256()
        for chunk in entry["chunks"]:
            self._file.seek(chunk["offset"])
            data = self._file.read(chunk["length"])
            if len(data) != chunk["length"]:
                raise NxsFormatError(f"truncated payload chunk in {name!r}")
            if verify and hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                raise NxsFormatError(f"payload chunk SHA-256 mismatch in {name!r}")
            whole_hash.update(data)
            yield data
        if verify and whole_hash.hexdigest() != entry["sha256"]:
            raise NxsFormatError(f"entry SHA-256 mismatch in {name!r}")

    def verify(self) -> None:
        for entry in self.entries:
            for _ in self.iter_entry(entry["name"], verify=True):
                pass

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "NxsArchive":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
