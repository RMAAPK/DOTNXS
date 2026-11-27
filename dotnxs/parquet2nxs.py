"""Pack Parquet dataset shards into one lossless NXS archive."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from .archive import DEFAULT_CHUNK_BYTES, NxsWriter


def _safe_relative_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe Parquet shard path: {relative!r}")
    return relative


def _inspect_parquet(path: Path) -> dict:
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"{path.name!r} is too short to be a Parquet file")
    with path.open("rb") as file_obj:
        if file_obj.read(4) != b"PAR1":
            raise ValueError(f"{path.name!r} does not have the Parquet PAR1 header")
        file_obj.seek(-4, os.SEEK_END)
        if file_obj.read(4) != b"PAR1":
            raise ValueError(f"{path.name!r} does not have the Parquet PAR1 footer")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet metadata inspection requires `pip install 'dotnxs[parquet]'`") from exc

    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    row_groups = []
    for index in range(metadata.num_row_groups):
        group = metadata.row_group(index)
        row_groups.append(
            {
                "index": index,
                "rows": group.num_rows,
                "uncompressed_bytes": group.total_byte_size,
                "compressed_bytes": sum(group.column(column).total_compressed_size for column in range(group.num_columns)),
            }
        )
    return {
        "size": size,
        "schema": parquet.schema_arrow.to_string(),
        "row_count": metadata.num_rows,
        "row_group_count": metadata.num_row_groups,
        "row_groups": row_groups,
    }


def _parquet_paths(source: Path) -> tuple[Path, list[Path]]:
    if source.is_file():
        if source.suffix.lower() != ".parquet":
            raise ValueError("input file must have a .parquet extension")
        return source.parent, [source]
    if not source.is_dir():
        raise NotADirectoryError(f"Parquet input path not found: {source}")
    paths = sorted(path for path in source.rglob("*.parquet") if path.is_file())
    if not paths:
        raise ValueError(f"no .parquet shards found in {source}")
    return source, paths


def convert_parquet(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    overwrite: bool = False,
    source_metadata: dict | None = None,
) -> dict:
    """Package one Parquet file or a directory of Parquet shards into NXS.

    Shard contents remain byte-for-byte Parquet. The converter records Arrow
    schema and row-group summaries in the NXS manifest, then streams payloads
    into aligned NXS chunks without reading whole shards into RAM.
    """
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination} (pass --force to replace it)")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be greater than zero")

    root, paths = _parquet_paths(source)
    if destination in paths:
        raise ValueError("output archive must not replace an input Parquet shard")

    shard_infos = []
    for path in paths:
        relative = _safe_relative_path(path, root)
        details = _inspect_parquet(path)
        shard_infos.append((path, relative, details))

    dataset_source = {
        "format": "parquet_dataset",
        "root_name": root.name,
        "shards": [
            {
                "name": f"dataset/{relative}",
                "source_path": relative,
                "size": details["size"],
                "row_count": details["row_count"],
                "row_group_count": details["row_group_count"],
            }
            for _path, relative, details in shard_infos
        ],
    }
    if source_metadata:
        dataset_source.update(source_metadata)

    with NxsWriter(
        destination,
        source=dataset_source,
        archive_metadata={
            "converter": "parquet2nxs",
            "converter_version": "0.2.0",
            "conversion": "lossless Parquet shard and metadata copy",
        },
        chunk_bytes=chunk_bytes,
    ) as writer:
        for path, relative, details in shard_infos:
            with path.open("rb") as shard_file:
                writer.add_stream(
                    f"dataset/{relative}",
                    shard_file,
                    details["size"],
                    kind="parquet_shard",
                    metadata={
                        "format": "parquet",
                        "source_path": relative,
                        "schema": details["schema"],
                        "row_count": details["row_count"],
                        "row_group_count": details["row_group_count"],
                        "row_groups": details["row_groups"],
                    },
                )

    return {
        "input": str(source),
        "output": str(destination),
        "parquet_files": len(shard_infos),
        "row_count": sum(details["row_count"] for _path, _relative, details in shard_infos),
        "row_group_count": sum(details["row_group_count"] for _path, _relative, details in shard_infos),
        "input_bytes": sum(details["size"] for _path, _relative, details in shard_infos),
        "chunk_bytes": chunk_bytes,
    }


__all__ = ["convert_parquet"]
