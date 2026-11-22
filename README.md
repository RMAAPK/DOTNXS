# DOTNXS (`.nxs`)

DOTNXS is a single-file archive format for AI/LLM artifacts. One `.nxs` file contains internal tensor, asset, and later dataset chunks, with an index intended to support selective reads and RAM-budgeted caching.

The project is converter-first. The CLI wraps a C++17/pybind11 native archive writer, while Python handles source-format metadata and conversion orchestration. The streaming Safetensors converter preserves tensor bytes; model runtime is not implemented yet. GitHub Actions builds and tests the native wheel on Linux, macOS, and Windows.

- [Format specification](FORMAT.md)
- [Python package and CLI](pyproject.toml)
- License: MIT (see [LICENSE](LICENSE))

## CLI

```bash
dotnxs --help
dotnxs convert safetensors input.safetensors output.nxs
dotnxs inspect output.nxs
dotnxs verify output.nxs
```

The default internal chunk size is 4 MiB and can be changed with `--chunk-mib`. The converter does not quantize weights. It currently accepts one local `.safetensors` file; Transformers-directory, Hugging Face Hub, GGUF, and Parquet converters are planned follow-ups. Model inference is not implemented yet.

Run the converter tests with:

```bash
python -m unittest discover -s tests -v
```

The current tests use a tiny generated fixture. A full Inkling 2 Mini conversion and inference run have **not** been performed yet.

The format's 20-byte signature is fixed. Its 16-byte suffix is a UUIDv4-derived value for a very low accidental-collision probability, not a claim of globally guaranteed uniqueness.
