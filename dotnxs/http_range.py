"""Seekable, bounded-memory reader backed by HTTP single-range requests."""

from __future__ import annotations

import re
from typing import Any

_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


class HTTPRangeError(OSError):
    """Raised when a remote server cannot satisfy an exact byte-range read."""


class HTTPRangeReader:
    """A small read-ahead cache over an HTTP resource with a known size.

    ``read(size)`` never downloads more than the configured read-ahead window
    and the requested bytes. Servers must honor a single HTTP Range request.
    """

    def __init__(
        self,
        url: str,
        size: int,
        *,
        headers: dict[str, str] | None = None,
        read_ahead_bytes: int = 32 * 1024 * 1024,
        timeout: tuple[float, float] = (15.0, 120.0),
        session: Any | None = None,
    ) -> None:
        if size < 0:
            raise ValueError("resource size must be non-negative")
        if read_ahead_bytes <= 0:
            raise ValueError("read_ahead_bytes must be greater than zero")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("HTTP range reads require requests (install dotnxs[hf])") from exc

        self.url = url
        self.size = size
        self.read_ahead_bytes = read_ahead_bytes
        self.timeout = timeout
        self.session = session or requests.Session()
        self._owns_session = session is None
        self.headers = {"Accept-Encoding": "identity", "User-Agent": "dotnxs-http-range/0.2"}
        if headers:
            self.headers.update(headers)
        self._position = 0
        self._cache_start = 0
        self._cache = b""
        self._closed = False

    def tell(self) -> int:
        self._check_open()
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        self._check_open()
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence value: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._check_open()
        if size is None or size < 0:
            raise ValueError("unbounded reads are disabled; request a specific byte count")
        if size == 0 or self._position >= self.size:
            return b""

        remaining = min(size, self.size - self._position)
        result = bytearray()
        while remaining:
            cache_end = self._cache_start + len(self._cache)
            if not (self._cache_start <= self._position < cache_end):
                self._fetch_range(remaining)
                cache_end = self._cache_start + len(self._cache)
            cache_offset = self._position - self._cache_start
            count = min(remaining, cache_end - self._position)
            if count <= 0:
                raise HTTPRangeError("server returned an empty or unusable byte range")
            result.extend(self._cache[cache_offset : cache_offset + count])
            self._position += count
            remaining -= count
        return bytes(result)

    def _fetch_range(self, minimum_bytes: int) -> None:
        start = self._position
        end_exclusive = min(self.size, start + max(self.read_ahead_bytes, minimum_bytes))
        end = end_exclusive - 1
        request_headers = dict(self.headers)
        request_headers["Range"] = f"bytes={start}-{end}"
        response = self.session.get(
            self.url,
            headers=request_headers,
            timeout=self.timeout,
            allow_redirects=True,
        )
        try:
            if response.status_code != 206:
                response.raise_for_status()
                raise HTTPRangeError(
                    f"server did not honor Range request (expected 206, got {response.status_code})"
                )
            content_range = response.headers.get("Content-Range", "")
            match = _CONTENT_RANGE.match(content_range)
            if not match:
                raise HTTPRangeError("server returned a malformed Content-Range header")
            actual_start, actual_end = int(match.group(1)), int(match.group(2))
            total_text = match.group(3)
            if actual_start != start or actual_end > end or actual_end < actual_start:
                raise HTTPRangeError(f"unexpected Content-Range: {content_range!r}")
            if total_text != "*" and int(total_text) != self.size:
                raise HTTPRangeError(
                    f"remote size changed during read (expected {self.size}, got {total_text})"
                )
            data = response.content
            expected_length = actual_end - actual_start + 1
            if len(data) != expected_length:
                raise HTTPRangeError(
                    f"range length mismatch: expected {expected_length}, received {len(data)}"
                )
            self._cache_start = actual_start
            self._cache = data
        finally:
            response.close()

    def clear_cache(self) -> None:
        """Drop the single read-ahead buffer without closing the HTTP reader."""
        self._check_open()
        self._cache_start = self._position
        self._cache = b""

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("I/O operation on closed HTTPRangeReader")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._cache = b""
            if self._owns_session:
                self.session.close()

    def __enter__(self) -> "HTTPRangeReader":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
