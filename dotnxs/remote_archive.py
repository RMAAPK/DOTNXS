"""Remote NXS archive reader with verified, size-bounded chunk caching."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from typing import Iterator

from .archive import ALIGNMENT, HEADER_SIZE, HEADER_STRUCT, MAGIC, MAX_MANIFEST_BYTES, NXS_VERSION, NxsFormatError
from .http_range import HTTPRangeReader

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CACHE_BYTES = 200 * 1024 * 1024


class RemoteNxsArchive:
    """Read internal NXS entries by HTTP byte range without downloading the archive.

    The cache is an upper bound for verified NXS payload chunks. It is lazy: no
    cache memory is reserved until a chunk is requested. A chunk larger than the
    configured cap is returned to the caller but is not retained in the cache.
    """

    def __init__(
        self,
        url: str,
        file_size: int,
        *,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(file_size, int) or file_size < HEADER_SIZE:
            raise ValueError("file_size must be at least the fixed NXS header size")
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        self.url = url
        self.file_size = file_size
        self.cache_limit_bytes = cache_bytes
        self.cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.network_bytes = 0
        self.network_ranges = 0
        self._chunk_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._reader = HTTPRangeReader(
            url,
            file_size,
            headers=headers,
            read_ahead_bytes=1,
        )
        try:
            self._reader.seek(0)
            raw_header = self._reader.read(HEADER_SIZE)
            self.network_bytes += len(raw_header)
            self.network_ranges += 1
            if len(raw_header) != HEADER_SIZE:
                raise NxsFormatError("truncated remote NXS header")
            (
                magic,
                major,
                minor,
                header_size,
                flags,
                manifest_offset,
                manifest_length,
                expected_manifest_digest,
            ) = HEADER_STRUCT.unpack(raw_header)
            if magic != MAGIC:
                raise NxsFormatError("remote archive has the wrong DOTNXS magic")
            if (major, minor) != NXS_VERSION or header_size != HEADER_SIZE or flags != 0:
                raise NxsFormatError("unsupported or invalid remote NXS header")
            if (
                manifest_length <= 0
                or manifest_length > MAX_MANIFEST_BYTES
                or manifest_offset < HEADER_SIZE
                or manifest_offset + manifest_length != file_size
            ):
                raise NxsFormatError("remote NXS manifest range is invalid")

            self._reader.seek(manifest_offset)
            manifest_bytes = self._reader.read(manifest_length)
            self.network_bytes += len(manifest_bytes)
            self.network_ranges += 1
            if len(manifest_bytes) != manifest_length:
                raise NxsFormatError("truncated remote NXS manifest")
            if hashlib.sha256(manifest_bytes).digest() != expected_manifest_digest:
                raise NxsFormatError("remote NXS manifest SHA-256 mismatch")
            try:
                self.manifest = json.loads(manifest_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NxsFormatError("remote NXS manifest is invalid JSON") from exc
            if self.manifest.get("format") != "DOTNXS" or self.manifest.get("version") != list(NXS_VERSION):
                raise NxsFormatError("remote NXS manifest format/version mismatch")
            self.manifest_offset = manifest_offset
            self.entries = self.manifest.get("entries")
            self._entry_map = self._validate_entries()
        except Exception:
            self._reader.close()
            raise

    @classmethod
    def from_hf_repo(
        cls,
        repo_id: str,
        filename: str,
        *,
        revision: str = "main",
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        token: str | None = None,
    ) -> "RemoteNxsArchive":
        """Open an NXS file hosted on the Hugging Face Hub by repo and path."""
        try:
            from huggingface_hub import HfApi, hf_hub_url
        except ImportError as exc:
            raise RuntimeError("HF repo lookup requires `huggingface_hub`") from exc
        api = HfApi(token=token)
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        file_info = next((item for item in info.siblings or [] if item.rfilename == filename), None)
        if file_info is None:
            raise FileNotFoundError(f"{filename!r} was not found in {repo_id}@{revision}")
        file_size = getattr(file_info, "size", None)
        if file_size is None:
            lfs_info = getattr(file_info, "lfs", None)
            file_size = lfs_info.get("size") if isinstance(lfs_info, dict) else getattr(lfs_info, "size", None)
        if not isinstance(file_size, int):
            raise ValueError(f"Hub did not report a file size for {filename!r}")
        resolved_revision = info.sha or revision
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=resolved_revision,
        )
        auth_headers = {"Authorization": f"Bearer {token}"} if token else None
        archive = cls(url, file_size, cache_bytes=cache_bytes, headers=auth_headers)
        archive.source_repo = repo_id
        archive.source_revision = resolved_revision
        archive.source_filename = filename
        return archive

    def _validate_entries(self) -> dict[str, dict]:
        if not isinstance(self.entries, list):
            raise NxsFormatError("remote manifest entries must be a list")
        entry_map: dict[str, dict] = {}
        ranges: list[tuple[int, int, str]] = []
        for entry in self.entries:
            if not isinstance(entry, dict):
                raise NxsFormatError("remote manifest entry must be an object")
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in entry_map:
                raise NxsFormatError("remote manifest has an invalid or duplicate entry name")
            size = entry.get("nbytes")
            digest = entry.get("sha256")
            chunks = entry.get("chunks")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise NxsFormatError(f"invalid remote entry size for {name!r}")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise NxsFormatError(f"invalid remote entry digest for {name!r}")
            if not isinstance(chunks, list):
                raise NxsFormatError(f"invalid remote chunk list for {name!r}")
            logical_position = 0
            for chunk in chunks:
                offset = chunk.get("offset") if isinstance(chunk, dict) else None
                length = chunk.get("length") if isinstance(chunk, dict) else None
                logical = chunk.get("logical_offset") if isinstance(chunk, dict) else None
                chunk_hash = chunk.get("sha256") if isinstance(chunk, dict) else None
                values = (offset, length, logical)
                if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
                    raise NxsFormatError(f"invalid remote chunk range for {name!r}")
                if length == 0 or logical != logical_position:
                    raise NxsFormatError(f"non-contiguous logical chunks for {name!r}")
                if offset < HEADER_SIZE or offset % ALIGNMENT != 0 or offset + length > self.manifest_offset:
                    raise NxsFormatError(f"remote chunk outside payload area for {name!r}")
                if not isinstance(chunk_hash, str) or not _SHA256_RE.fullmatch(chunk_hash):
                    raise NxsFormatError(f"invalid remote chunk digest for {name!r}")
                logical_position += length
                ranges.append((offset, offset + length, name))
            if logical_position != size:
                raise NxsFormatError(f"remote chunks do not cover entry {name!r}")
            entry_map[name] = entry

        ranges.sort()
        previous_end = HEADER_SIZE
        for start, end, name in ranges:
            if start < previous_end:
                raise NxsFormatError(f"overlapping remote chunks near {name!r}")
            previous_end = end
        return entry_map

    def list_entries(self, *, kind: str | None = None) -> list[dict]:
        if kind is None:
            return list(self.entries)
        return [entry for entry in self.entries if entry.get("kind") == kind]

    def get_entry(self, name: str) -> dict:
        try:
            return self._entry_map[name]
        except KeyError as exc:
            raise KeyError(f"NXS entry not found: {name}") from exc

    def _get_chunk(self, name: str, index: int, chunk: dict) -> bytes:
        key = (name, index)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            self._chunk_cache.move_to_end(key)
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        self._reader.seek(chunk["offset"])
        data = self._reader.read(chunk["length"])
        self.network_bytes += len(data)
        self.network_ranges += 1
        if len(data) != chunk["length"]:
            raise NxsFormatError(f"truncated remote payload chunk in {name!r}")
        if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
            raise NxsFormatError(f"remote payload chunk SHA-256 mismatch in {name!r}")
        if len(data) <= self.cache_limit_bytes:
            while self._chunk_cache and self.cache_bytes + len(data) > self.cache_limit_bytes:
                _, oldest = self._chunk_cache.popitem(last=False)
                self.cache_bytes -= len(oldest)
                self.cache_evictions += 1
            self._chunk_cache[key] = data
            self.cache_bytes += len(data)
        return data

    def iter_entry_range(self, name: str, start: int = 0, length: int | None = None) -> Iterator[bytes]:
        entry = self.get_entry(name)
        total_size = entry["nbytes"]
        if length is None:
            length = total_size - start
        if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length < 0:
            raise ValueError("entry range start and length must be non-negative integers")
        if start > total_size or length > total_size - start:
            raise ValueError("requested entry range exceeds the entry size")
        if length == 0:
            return
        end = start + length
        for index, chunk in enumerate(entry["chunks"]):
            logical_start = chunk["logical_offset"]
            logical_end = logical_start + chunk["length"]
            overlap_start = max(start, logical_start)
            overlap_end = min(end, logical_end)
            if overlap_start >= overlap_end:
                continue
            data = self._get_chunk(name, index, chunk)
            left = overlap_start - logical_start
            right = overlap_end - logical_start
            yield data[left:right]

    def read_entry_range(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        return b"".join(self.iter_entry_range(name, start, length))

    def read_entry(self, name: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
        entry = self.get_entry(name)
        if entry["nbytes"] > max_bytes:
            raise ValueError(f"entry {name!r} exceeds the read_entry limit; use iter_entry_range")
        return self.read_entry_range(name, 0, entry["nbytes"])

    def read_tensor_expert(self, name: str, expert_id: int, n_experts: int) -> bytes:
        entry = self.get_entry(name)
        shape = entry.get("shape")
        if entry.get("kind") != "tensor" or not isinstance(shape, list) or not shape:
            raise ValueError(f"entry {name!r} is not a tensor with an expert-leading dimension")
        if shape[0] != n_experts or not 0 <= expert_id < n_experts:
            raise ValueError(f"expert index {expert_id} or expert count {n_experts} does not match {name!r}")
        if entry["nbytes"] % n_experts != 0:
            raise NxsFormatError(f"tensor byte size is not evenly divisible by expert count for {name!r}")
        expert_bytes = entry["nbytes"] // n_experts
        return self.read_entry_range(name, expert_id * expert_bytes, expert_bytes)

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "limit_bytes": self.cache_limit_bytes,
            "used_bytes": self.cache_bytes,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "evictions": self.cache_evictions,
            "network_bytes": self.network_bytes,
            "network_ranges": self.network_ranges,
        }

    def close(self) -> None:
        self._reader.close()
        self._chunk_cache.clear()
        self.cache_bytes = 0

    def __enter__(self) -> "RemoteNxsArchive":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
