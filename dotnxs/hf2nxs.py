"""Convert a Hugging Face Transformers repo to one NXS archive using HTTP ranges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from .archive import DEFAULT_CHUNK_BYTES, NxsWriter
from .http_range import HTTPRangeReader
from .safetensors2nxs import _validate_header, read_safetensors_header


def _sibling_size(sibling) -> int:
    size = getattr(sibling, "size", None)
    if size is None:
        lfs = getattr(sibling, "lfs", None)
        size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"Hub did not provide a valid file size for {sibling.rfilename!r}")
    return size


def _safe_name(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe path in Hugging Face repo: {name!r}")
    return path.as_posix()


def _read_asset(reader: HTTPRangeReader, size: int, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    if size > max_bytes:
        raise ValueError(f"metadata asset is unexpectedly large ({size} bytes)")
    reader.seek(0)
    data = reader.read(size)
    if len(data) != size:
        raise OSError("truncated remote metadata asset")
    return data


def convert_repo(
    repo_id: str,
    output_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    overwrite: bool = False,
    read_ahead_bytes: int = 32 * 1024 * 1024,
) -> dict:
    """Build one .nxs archive by fetching only required source byte ranges.

    The source snapshot is not downloaded to disk. Each Safetensors tensor and
    model asset is fetched with single-range HTTP requests and streamed into the
    NXS payload. This keeps peak RAM bounded by the NXS chunk and HTTP read-ahead
    buffers, rather than by the full checkpoint size.
    """
    if not isinstance(repo_id, str) or "/" not in repo_id or repo_id.startswith("/"):
        raise ValueError("repo_id must be in owner/name form")
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination} (pass --force to replace it)")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be greater than zero")
    if read_ahead_bytes <= 0:
        raise ValueError("read_ahead_bytes must be greater than zero")

    try:
        import requests
        from huggingface_hub import HfApi, get_token, hf_hub_url
    except ImportError as exc:
        raise RuntimeError(
            "HF conversion requires the optional dependency; install with "
            "`python -m pip install 'dotnxs[hf]'`"
        ) from exc

    token = get_token()
    api = HfApi(token=token)
    repo_info = api.model_info(repo_id, revision=revision, files_metadata=True)
    resolved_revision = repo_info.sha or revision or "main"
    siblings = sorted(repo_info.siblings or [], key=lambda item: item.rfilename)
    file_sizes = {_safe_name(item.rfilename): _sibling_size(item) for item in siblings}
    if not file_sizes:
        raise ValueError(f"Hugging Face repo {repo_id!r} contains no files")

    session = requests.Session()
    headers = {"User-Agent": "dotnxs-hf2nxs/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def make_reader(filename: str) -> HTTPRangeReader:
        safe_filename = _safe_name(filename)
        url = hf_hub_url(
            repo_id=repo_id,
            filename=safe_filename,
            repo_type="model",
            revision=resolved_revision,
        )
        return HTTPRangeReader(
            url,
            file_sizes[safe_filename],
            headers=headers,
            read_ahead_bytes=read_ahead_bytes,
            session=session,
        )

    readers: dict[str, HTTPRangeReader] = {}
    try:
        weight_names = [name for name in file_sizes if name.lower().endswith(".safetensors")]
        if not weight_names:
            raise ValueError(f"Hugging Face repo {repo_id!r} has no Safetensors checkpoints")

        infos: list[dict] = []
        files_by_name: dict[str, set[str]] = {}
        for filename in weight_names:
            reader = make_reader(filename)
            readers[filename] = reader
            header, data_start = read_safetensors_header(reader, file_sizes[filename])
            tensor_records, metadata = _validate_header(header, file_sizes[filename] - data_start)
            info = {
                "relative_path": filename,
                "size": file_sizes[filename],
                "data_start": data_start,
                "tensors": tensor_records,
                "metadata": metadata,
            }
            infos.append(info)
            files_by_name[filename] = {record[1] for record in tensor_records}

        index_name = "model.safetensors.index.json"
        if index_name in file_sizes:
            index_reader = readers.get(index_name) or make_reader(index_name)
            readers[index_name] = index_reader
            index_bytes = _read_asset(index_reader, file_sizes[index_name])
            try:
                index_data = json.loads(index_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid model.safetensors.index.json") from exc
            _validate_weight_index_data(index_data, files_by_name)

        tensor_names: set[str] = set()
        for info in infos:
            prefix = "mtp/" if "mtp" in Path(info["relative_path"]).stem.lower() else ""
            for _, tensor_name, _ in info["tensors"]:
                archive_name = prefix + tensor_name
                if archive_name in tensor_names:
                    raise ValueError(f"duplicate tensor name across Safetensors files: {archive_name!r}")
                tensor_names.add(archive_name)

        model_config = {}
        if "config.json" in file_sizes:
            config_reader = readers.get("config.json") or make_reader("config.json")
            readers["config.json"] = config_reader
            try:
                model_config = json.loads(_read_asset(config_reader, file_sizes["config.json"]).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid config.json in Hugging Face repo") from exc
            if not isinstance(model_config, dict):
                raise ValueError("config.json must contain a JSON object")

        file_descriptions = [
            {
                "name": name,
                "kind": "safetensors" if name.lower().endswith(".safetensors") else "asset",
                "size": file_sizes[name],
                **next(
                    (
                        {
                            "tensor_count": len(info["tensors"]),
                            "metadata": info["metadata"],
                        }
                        for info in infos
                        if info["relative_path"] == name
                    ),
                    {},
                ),
            }
            for name in file_sizes
        ]
        source = {
            "format": "transformers",
            "repo_id": repo_id,
            "requested_revision": revision or "main",
            "resolved_revision": resolved_revision,
            "model_type": model_config.get("model_type"),
            "architectures": model_config.get("architectures"),
            "files": file_descriptions,
        }
        archive_metadata = {
            "converter": "hf2nxs",
            "converter_version": "0.2.0.dev0",
            "conversion": "lossless tensor/asset copy via HTTP range reads",
        }

        with NxsWriter(
            destination,
            source=source,
            archive_metadata=archive_metadata,
            chunk_bytes=chunk_bytes,
        ) as writer:
            for info in infos:
                filename = info["relative_path"]
                reader = readers[filename]
                prefix = "mtp/" if "mtp" in Path(filename).stem.lower() else ""
                for begin, tensor_name, record in info["tensors"]:
                    data_begin, data_end = record["data_offsets"]
                    writer.add_stream(
                        prefix + tensor_name,
                        reader,
                        data_end - data_begin,
                        kind="tensor",
                        source_offset=info["data_start"] + data_begin,
                        dtype=record["dtype"],
                        shape=record["shape"],
                        metadata={"source_file": filename, "source_data_offset": data_begin},
                    )

            for filename, size in file_sizes.items():
                if filename.lower().endswith(".safetensors"):
                    continue
                reader = readers.get(filename)
                if reader is None:
                    reader = make_reader(filename)
                    readers[filename] = reader
                writer.add_stream(
                    f"asset/{filename}",
                    reader,
                    size,
                    kind="asset",
                    metadata={"source_file": filename},
                )

        return {
            "repo_id": repo_id,
            "revision": resolved_revision,
            "output": str(destination),
            "source_bytes": sum(file_sizes.values()),
            "tensor_count": len(tensor_names),
            "file_count": len(file_sizes),
            "chunk_bytes": chunk_bytes,
        }
    finally:
        for reader in readers.values():
            reader.close()
        session.close()


def _validate_weight_index_data(index_data: dict, files_by_name: dict[str, set[str]]) -> None:
    weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json must contain an object named 'weight_map'")
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(filename, str):
            raise ValueError("Safetensors weight_map keys and values must be strings")
        relative = _safe_name(filename)
        tensor_names = files_by_name.get(relative)
        if tensor_names is None:
            raise ValueError(f"weight_map references missing shard {relative!r}")
        if tensor_name not in tensor_names:
            raise ValueError(
                f"weight_map assigns tensor {tensor_name!r} to {relative!r}, "
                "but that tensor is absent from the shard header"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hf2nxs",
        description="Convert a Hugging Face repo into one NXS archive using HTTP range reads.",
    )
    parser.add_argument("repo_id", help="Hugging Face model repo, for example owner/model")
    parser.add_argument("output", help="destination single-file .nxs archive")
    parser.add_argument("--revision", help="branch, tag, or commit to convert (default: main)")
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_BYTES // (1024 * 1024))
    parser.add_argument("--read-ahead-mib", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    if args.chunk_mib <= 0 or args.read_ahead_mib <= 0:
        parser.error("--chunk-mib and --read-ahead-mib must be greater than zero")
    try:
        result = convert_repo(
            args.repo_id,
            args.output,
            revision=args.revision,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            read_ahead_bytes=args.read_ahead_mib * 1024 * 1024,
            overwrite=args.force,
        )
    except Exception as exc:
        parser.exit(2, f"hf2nxs: error: {exc}\n")
    print(
        f"Created {result['output']} from {result['repo_id']}@{result['revision']}: "
        f"{result['tensor_count']} tensors, fetched {result['source_bytes']} source bytes via ranges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
