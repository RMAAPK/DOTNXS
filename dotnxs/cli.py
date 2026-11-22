"""The `dotnxs` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from . import __version__
from .archive import NxsArchive
from .hf2nxs import convert_repo
from .safetensors2nxs import convert_file
from .transformers2nxs import convert_directory


def _package_version() -> str:
    return __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dotnxs",
        description="Create and inspect single-file DOTNXS archives.",
    )
    parser.add_argument("--version", action="version", version=f"dotnxs {_package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="convert a source checkpoint")
    convert_subparsers = convert_parser.add_subparsers(dest="source_format", required=True)
    safe_parser = convert_subparsers.add_parser("safetensors", help="convert one Safetensors file")
    safe_parser.add_argument("input", help="source .safetensors file")
    safe_parser.add_argument("output", help="destination single-file .nxs archive")
    safe_parser.add_argument("--chunk-mib", type=int, default=4)
    safe_parser.add_argument("--force", action="store_true")

    transformers_parser = convert_subparsers.add_parser(
        "transformers", help="convert a local Transformers model directory"
    )
    transformers_parser.add_argument("input", help="local model directory")
    transformers_parser.add_argument("output", help="destination .nxs archive")
    transformers_parser.add_argument("--chunk-mib", type=int, default=4)
    transformers_parser.add_argument("--force", action="store_true")

    hf_parser = convert_subparsers.add_parser("hf", help="download and convert a Hugging Face model repo")
    hf_parser.add_argument("repo_id", help="Hugging Face repo in owner/name form")
    hf_parser.add_argument("output", help="destination .nxs archive")
    hf_parser.add_argument("--revision", help="branch, tag, or commit (default: main)")
    hf_parser.add_argument("--cache-dir", help="Hugging Face Hub cache directory")
    hf_parser.add_argument("--chunk-mib", type=int, default=4)
    hf_parser.add_argument("--force", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="show an NXS archive's manifest summary")
    inspect_parser.add_argument("archive", help="path to a .nxs archive")

    verify_parser = subparsers.add_parser("verify", help="verify the manifest and all payload hashes")
    verify_parser.add_argument("archive", help="path to a .nxs archive")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "convert" and args.source_format in {"safetensors", "transformers", "hf"}:
            if args.chunk_mib <= 0:
                parser.error("--chunk-mib must be greater than zero")
            chunk_bytes = args.chunk_mib * 1024 * 1024
            if args.source_format == "safetensors":
                result = convert_file(
                    args.input,
                    args.output,
                    chunk_bytes=chunk_bytes,
                    overwrite=args.force,
                )
                print(
                    f"Created {result['output']}: {result['tensor_count']} tensors, "
                    f"{result['input_bytes']} input bytes"
                )
                return 0
            if args.source_format == "transformers":
                result = convert_directory(
                    args.input,
                    args.output,
                    chunk_bytes=chunk_bytes,
                    overwrite=args.force,
                )
                print(
                    f"Created {result['output']}: {result['tensor_count']} tensors from "
                    f"{result['weight_files']} weight files and {result['asset_files']} assets"
                )
                return 0
            result = convert_repo(
                args.repo_id,
                args.output,
                revision=args.revision,
                cache_dir=args.cache_dir,
                chunk_bytes=chunk_bytes,
                overwrite=args.force,
            )
            print(
                f"Created {result['output']} from {result['repo_id']}@{result['revision']}: "
                f"{result['tensor_count']} tensors"
            )
            return 0

        if args.command == "inspect":
            with NxsArchive(args.archive) as archive:
                summary = {
                    "format": archive.manifest["format"],
                    "version": archive.manifest["version"],
                    "source": archive.manifest.get("source", {}),
                    "entry_count": len(archive.entries),
                    "entries": [
                        {
                            "name": entry["name"],
                            "kind": entry["kind"],
                            "nbytes": entry["nbytes"],
                            "dtype": entry.get("dtype"),
                            "shape": entry.get("shape"),
                            "chunk_count": len(entry["chunks"]),
                        }
                        for entry in archive.entries
                    ],
                }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "verify":
            with NxsArchive(args.archive) as archive:
                archive.verify()
                count = len(archive.entries)
            print(f"OK: verified {count} entries in {args.archive}")
            return 0

        parser.error("unsupported command")
    except Exception as exc:
        print(f"dotnxs: error: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
