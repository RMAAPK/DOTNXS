"""Download a Hugging Face model snapshot and pack it into one NXS archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .archive import DEFAULT_CHUNK_BYTES
from .transformers2nxs import convert_directory


def convert_repo(
    repo_id: str,
    output_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    overwrite: bool = False,
) -> dict:
    """Fetch a Hub snapshot, then convert its local Transformers directory."""
    if not isinstance(repo_id, str) or "/" not in repo_id or repo_id.startswith("/"):
        raise ValueError("repo_id must be in owner/name form")
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination} (pass --force to replace it)")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "HF conversion requires the optional dependency; install with "
            "`python -m pip install 'dotnxs[hf]'`"
        ) from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
        )
    )
    resolved_revision = snapshot_path.name if snapshot_path.parent.name == "snapshots" else None
    source_metadata = {
        "repo_id": repo_id,
        "requested_revision": revision or "main",
        "resolved_revision": resolved_revision,
        "hub": "huggingface.co",
    }
    result = convert_directory(
        snapshot_path,
        destination,
        chunk_bytes=chunk_bytes,
        overwrite=overwrite,
        source_metadata=source_metadata,
    )
    result["repo_id"] = repo_id
    result["revision"] = resolved_revision or revision or "main"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hf2nxs",
        description="Download a Hugging Face Transformers repo and convert it into one NXS archive.",
    )
    parser.add_argument("repo_id", help="Hugging Face model repo, for example owner/model")
    parser.add_argument("output", help="destination single-file .nxs archive")
    parser.add_argument("--revision", help="branch, tag, or commit to download (default: main)")
    parser.add_argument("--cache-dir", help="Hugging Face Hub cache directory")
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_BYTES // (1024 * 1024))
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be greater than zero")
    try:
        result = convert_repo(
            args.repo_id,
            args.output,
            revision=args.revision,
            cache_dir=args.cache_dir,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            overwrite=args.force,
        )
    except Exception as exc:
        parser.exit(2, f"hf2nxs: error: {exc}\n")
    print(
        f"Created {result['output']} from {result['repo_id']}@{result['revision']}: "
        f"{result['tensor_count']} tensors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
