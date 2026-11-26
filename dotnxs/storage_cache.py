"""Bounded, checksum-verified file cache for memory-mapped tensor weights."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_RESERVE_BYTES = 256 * 1024 * 1024


class StorageCacheError(RuntimeError):
    """Raised when an entry cannot be safely materialized in the storage cache."""


class FileBackedTensorCache:
    """Materialize NXS entries as bounded disk files for CPU memory mapping.

    Entries are written to temporary files and atomically renamed only after
    their byte count and manifest SHA-256 match. The cache is an LRU capped by
    both the configured byte limit and currently available filesystem space.
    It stores files only; callers map them with ``torch.from_file`` or
    ``numpy.memmap`` rather than copying a full tensor into a Python buffer.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_bytes: int,
        reserve_bytes: int = _DEFAULT_RESERVE_BYTES,
    ) -> None:
        if not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if not isinstance(reserve_bytes, int) or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a non-negative integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._configured_limit_bytes = max_bytes
        self._reserve_bytes = reserve_bytes
        self.limit_bytes = max_bytes
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._files: OrderedDict[str, tuple[Path, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._scan()

    def _scan(self) -> None:
        files = []
        for path in self.directory.glob("*.tensor"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                files.append((stat.st_mtime_ns, path.stem, path, stat.st_size))
            except OSError:
                continue
        for _mtime, key, path, size in sorted(files):
            self._files[key] = (path, size)
            self.used_bytes += size
        free_bytes = shutil.disk_usage(self.directory).free
        self.limit_bytes = min(
            self._configured_limit_bytes,
            max(0, free_bytes + self.used_bytes - self._reserve_bytes),
        )
        self._evict_until_fits(0)
        for path in self.directory.glob("*.part"):
            try:
                path.unlink()
            except OSError:
                pass

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        skipped = 0
        while self._files and self.used_bytes + incoming_bytes > self.limit_bytes:
            key, (path, size) = self._files.popitem(last=False)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                self._files[key] = (path, size)
                skipped += 1
                if skipped >= len(self._files):
                    raise StorageCacheError(
                        "storage cache is full and its oldest files are still mapped or locked"
                    )
                continue
            self.used_bytes -= size
            self.evictions += 1
            skipped = 0

    def materialize(
        self,
        entry: dict,
        chunks: Iterable[bytes],
        *,
        cache_key: str | None = None,
    ) -> Path:
        """Return a local file; verify the full entry hash when it is available.

        ``cache_key`` supports logical slices (for example one expert within an
        expert-leading tensor) whose independent digest is not in the NXS manifest.
        Those slices are still assembled from individually SHA-256-verified NXS chunks.
        """
        expected_digest = entry.get("sha256")
        size = entry.get("nbytes")
        if expected_digest is not None and (
            not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(expected_digest)
        ):
            raise ValueError("entry sha256 must be a lowercase SHA-256 digest")
        key = cache_key or expected_digest
        if not isinstance(key, str) or not _SHA256_RE.fullmatch(key):
            raise ValueError("a lowercase SHA-256 digest or cache_key is required")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("entry must have a non-negative integer byte size")
        if size > self.limit_bytes:
            raise StorageCacheError(
                f"entry is {size} bytes, larger than the available storage-cache limit "
                f"of {self.limit_bytes} bytes"
            )

        destination = self.directory / f"{key}.tensor"
        with self._lock:
            existing = self._files.get(key)
            if existing is not None:
                existing_path, existing_size = existing
                if existing_path.is_file() and existing_size == size:
                    self._files.move_to_end(key)
                    self.hits += 1
                    try:
                        os.utime(existing_path, None)
                    except OSError:
                        pass
                    return existing_path
                try:
                    existing_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise StorageCacheError("cached tensor file is invalid but cannot be replaced") from exc
                self._files.pop(key, None)
                self.used_bytes -= existing_size
            self.misses += 1
            self._evict_until_fits(size)

            temp_fd, temp_name = tempfile.mkstemp(prefix=".nxs-", suffix=".part", dir=self.directory)
            temp_path = Path(temp_name)
            written = 0
            hasher = hashlib.sha256()
            try:
                with os.fdopen(temp_fd, "wb") as output:
                    for chunk in chunks:
                        view = memoryview(chunk)
                        written += len(view)
                        if written > size:
                            raise StorageCacheError("NXS entry stream exceeded its declared byte size")
                        output.write(view)
                        hasher.update(view)
                    output.flush()
                    os.fsync(output.fileno())
                if written != size:
                    raise StorageCacheError(
                        f"NXS entry stream ended at {written} bytes; expected {size}"
                    )
                actual_digest = hasher.hexdigest()
                if expected_digest is not None and actual_digest != expected_digest:
                    raise StorageCacheError("NXS entry SHA-256 does not match its manifest")
                os.replace(temp_path, destination)
                self._files[key] = (destination, size)
                self.used_bytes += size
                return destination
            except Exception:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
                raise

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit_bytes": self.limit_bytes,
                "used_bytes": self.used_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


__all__ = ["FileBackedTensorCache", "StorageCacheError"]
