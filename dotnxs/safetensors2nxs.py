"""Losslessly wrap a Safetensors file in one streaming DOTNXS archive."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

from .archive import DEFAULT_CHUNK_BYTES, NxsWriter

MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024


def read_safetensors_header(file_obj, file_size: int) -> tuple[dict, int]:
    """Return the JSON header and absolute offset of the tensor data section."""
    prefix = file_obj.read(8)
    if len(prefix) != 8:
        raise ValueError("input is too small to be a Safetensors file")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 1 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"implausible Safetensors header length: {header_length}")
    data_start = 8 + header_length
    if data_start > file_size:
        raise ValueError("Safetensors header extends beyond end of file")
    raw_header = file_obj.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError("truncated Safetensors JSON header")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    return header, data_start


def _validate_header(header: dict, data_size: int) -> tuple[list[tuple[int, str, dict]], dict]:
    safe_metadata = header.get("__metadata__", {})
    if not isinstance(safe_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in safe_metadata.items()
    ):
        raise ValueError("Safetensors __metadata__ must contain string keys and values")

    tensors: list[tuple[int, str, dict]] = []
    ranges: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("invalid Safetensors tensor record")
        dtype = record.get("dtype")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError(f"invalid dtype/shape metadata for tensor {name!r}")
        if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
            raise ValueError(f"invalid tensor shape for {name!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in offsets)
        ):
            raise ValueError(f"invalid data offsets for tensor {name!r}")
        start, end = offsets
        if end < start or end > data_size:
            raise ValueError(f"tensor range is outside the data section: {name!r}")
        record_copy = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        tensors.append((start, name, record_copy))
        if end > start:
            ranges.append((start, end, name))

    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start < previous_end:
            raise ValueError(f"overlapping Safetensors payload ranges near tensor {name!r}")
        previous_end = end
    tensors.sort(key=lambda item: (item[0], item[1]))
    return tensors, safe_metadata


def convert_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    overwrite: bool = False,
) -> dict:
    """Convert one .safetensors file to one .nxs archive, without loading weights into RAM."""
    source_path = Path(input_path)
    destination_path = Path(output_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("input and output paths must differ")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination_path} (pass --force to replace it)")
    file_size = source_path.stat().st_size

    with open(source_path, "rb") as source_file:
        header, data_start = read_safetensors_header(source_file, file_size)
        tensor_records, safe_metadata = _validate_header(header, file_size - data_start)
        tensor_count = len(tensor_records)
        source_description = {
            "format": "safetensors",
            "files": [
                {
                    "name": source_path.name,
                    "size": file_size,
                    "tensor_count": tensor_count,
                    "metadata": safe_metadata,
                }
            ],
        }
        archive_metadata = {
            "converter": "safetensors2nxs",
            "converter_version": "0.2.0",
            "conversion": "lossless byte-preserving tensor payload copy",
        }
        with NxsWriter(
            destination_path,
            source=source_description,
            archive_metadata=archive_metadata,
            chunk_bytes=chunk_bytes,
        ) as writer:
            for start, name, record in tensor_records:
                begin, end = record["data_offsets"]
                writer.add_stream(
                    name,
                    source_file,
                    end - begin,
                    kind="tensor",
                    source_offset=data_start + begin,
                    dtype=record["dtype"],
                    shape=record["shape"],
                    metadata={"source_data_offset": begin},
                )

    return {
        "input": str(source_path),
        "output": str(destination_path),
        "input_bytes": file_size,
        "tensor_count": tensor_count,
        "chunk_bytes": chunk_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safetensors2nxs",
        description="Convert one Safetensors checkpoint into a single DOTNXS archive.",
    )
    parser.add_argument("input", help="source .safetensors file")
    parser.add_argument("output", help="destination .nxs file")
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=DEFAULT_CHUNK_BYTES // (1024 * 1024),
        help="maximum bytes per internal NXS chunk in MiB (default: 4)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be greater than zero")
    try:
        result = convert_file(
            args.input,
            args.output,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            overwrite=args.force,
        )
    except Exception as exc:
        parser.exit(2, f"safetensors2nxs: error: {exc}\n")
    print(
        f"Created one NXS archive: {result['output']} "
        f"({result['tensor_count']} tensors, source {result['input_bytes']} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
