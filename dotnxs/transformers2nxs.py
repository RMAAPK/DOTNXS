"""Convert a local Hugging Face Transformers model directory to one NXS archive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from .archive import DEFAULT_CHUNK_BYTES, NxsWriter
from .safetensors2nxs import _validate_header, read_safetensors_header


SKIP_DIRECTORY_NAMES = {".git", ".cache", "__pycache__"}
UNSUPPORTED_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt"}


def _safe_relative_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe path in model directory: {relative!r}")
    return relative


def _inspect_safetensors(path: Path, relative_path: str) -> dict:
    file_size = path.stat().st_size
    with open(path, "rb") as file_obj:
        header, data_start = read_safetensors_header(file_obj, file_size)
    tensor_records, metadata = _validate_header(header, file_size - data_start)
    return {
        "path": path,
        "relative_path": relative_path,
        "size": file_size,
        "data_start": data_start,
        "tensors": tensor_records,
        "metadata": metadata,
    }


def _validate_weight_index(root: Path, infos: list[dict], files_by_name: dict[str, set[str]]) -> None:
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {index_path.name}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json must contain an object named 'weight_map'")

    for tensor_name, file_name in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(file_name, str):
            raise ValueError("Safetensors weight_map keys and values must be strings")
        relative = PurePosixPath(file_name.replace("\\", "/"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe shard path in weight_map: {file_name!r}")
        relative_name = relative.as_posix()
        tensor_names = files_by_name.get(relative_name)
        if tensor_names is None:
            raise ValueError(f"weight_map references missing shard {relative_name!r}")
        if tensor_name not in tensor_names:
            raise ValueError(
                f"weight_map assigns tensor {tensor_name!r} to {relative_name!r}, "
                "but that tensor is absent from the shard header"
            )


def convert_directory(
    input_directory: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    overwrite: bool = False,
    source_metadata: dict | None = None,
) -> dict:
    """Package Safetensors weights and model assets from a Transformers directory.

    Tensor payloads and assets are streamed into one .nxs file. Weights are
    preserved byte-for-byte; they are not deserialized or quantized.
    """
    root = Path(input_directory).resolve()
    destination = Path(output_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Transformers model directory not found: {root}")
    if destination.is_relative_to(root):
        raise ValueError("output .nxs archive must be outside the input model directory")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination} (pass --force to replace it)")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be greater than zero")

    all_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in SKIP_DIRECTORY_NAMES for part in path.relative_to(root).parts)
    )
    weight_paths = [path for path in all_files if path.suffix.lower() == ".safetensors"]
    if not weight_paths:
        unsupported = [
            path.relative_to(root).as_posix()
            for path in all_files
            if path.suffix.lower() in UNSUPPORTED_WEIGHT_SUFFIXES
        ]
        suffix = f"; found unsupported weight files: {unsupported}" if unsupported else ""
        raise ValueError(f"no Safetensors checkpoint files found in {root}{suffix}")

    infos = [_inspect_safetensors(path, _safe_relative_path(path, root)) for path in weight_paths]
    files_by_name: dict[str, set[str]] = {
        info["relative_path"]: {record[1] for record in info["tensors"]}
        for info in infos
    }
    _validate_weight_index(root, infos, files_by_name)

    tensor_names: set[str] = set()
    for info in infos:
        prefix = "mtp/" if "mtp" in Path(info["relative_path"]).stem.lower() else ""
        for _, tensor_name, _ in info["tensors"]:
            archive_name = prefix + tensor_name
            if archive_name in tensor_names:
                raise ValueError(f"duplicate tensor name across Safetensors files: {archive_name!r}")
            tensor_names.add(archive_name)

    asset_paths = [path for path in all_files if path.suffix.lower() != ".safetensors"]
    file_descriptions = [
        {
            "name": info["relative_path"],
            "kind": "safetensors",
            "size": info["size"],
            "tensor_count": len(info["tensors"]),
            "metadata": info["metadata"],
        }
        for info in infos
    ]
    file_descriptions.extend(
        {
            "name": _safe_relative_path(path, root),
            "kind": "asset",
            "size": path.stat().st_size,
        }
        for path in asset_paths
    )

    config_path = root / "config.json"
    model_config = {}
    if config_path.is_file():
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not parse config.json: {exc}") from exc
        if not isinstance(model_config, dict):
            raise ValueError("config.json must contain a JSON object")

    source = {
        "format": "transformers",
        "root_name": root.name,
        "model_type": model_config.get("model_type"),
        "architectures": model_config.get("architectures"),
        "files": file_descriptions,
    }
    if source_metadata:
        source.update(source_metadata)
    archive_metadata = {
        "converter": "transformers2nxs",
        "converter_version": "0.2.0",
        "conversion": "lossless Safetensors payload and asset copy",
    }

    with NxsWriter(
        destination,
        source=source,
        archive_metadata=archive_metadata,
        chunk_bytes=chunk_bytes,
    ) as writer:
        for info in infos:
            prefix = "mtp/" if "mtp" in Path(info["relative_path"]).stem.lower() else ""
            with open(info["path"], "rb") as source_file:
                for begin, tensor_name, record in info["tensors"]:
                    data_begin, data_end = record["data_offsets"]
                    writer.add_stream(
                        prefix + tensor_name,
                        source_file,
                        data_end - data_begin,
                        kind="tensor",
                        source_offset=info["data_start"] + data_begin,
                        dtype=record["dtype"],
                        shape=record["shape"],
                        metadata={
                            "source_file": info["relative_path"],
                            "source_data_offset": data_begin,
                        },
                    )
        for path in asset_paths:
            relative_name = _safe_relative_path(path, root)
            with open(path, "rb") as asset_file:
                writer.add_stream(
                    f"asset/{relative_name}",
                    asset_file,
                    path.stat().st_size,
                    kind="asset",
                    metadata={"source_file": relative_name},
                )

    return {
        "input": str(root),
        "output": str(destination),
        "weight_files": len(infos),
        "asset_files": len(asset_paths),
        "tensor_count": len(tensor_names),
        "chunk_bytes": chunk_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transformers2nxs",
        description="Convert a local Transformers model directory into one NXS archive.",
    )
    parser.add_argument("input", help="local Transformers model directory")
    parser.add_argument("output", help="destination single-file .nxs archive")
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_BYTES // (1024 * 1024))
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be greater than zero")
    try:
        result = convert_directory(
            args.input,
            args.output,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            overwrite=args.force,
        )
    except Exception as exc:
        parser.exit(2, f"transformers2nxs: error: {exc}\n")
    print(
        f"Created {result['output']}: {result['tensor_count']} tensors from "
        f"{result['weight_files']} Safetensors files and {result['asset_files']} assets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
