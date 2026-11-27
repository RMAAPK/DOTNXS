# DOTNXS `.nxs` archive format

**Status:** Draft implementation format

**Version:** 0.2

**Byte order:** Little-endian for numeric fields

An `.nxs` is **one archive file**. Model tensors, tokenizer/config assets, and later dataset records are entries inside it. Internal chunks are byte ranges within that single file; they are not separate `.nxs` files. The index lets a reader fetch or cache selected ranges without extracting the whole archive.

Version 0.2 is a lossless container. It does not quantize, decompress, or reinterpret imported tensor bytes. Model execution and dataset semantics are defined by the source model/asset and its metadata, not by this container format.

## File signature (magic)

Every `.nxs` file begins with these exact 20 bytes:

```text
4E 58 53 00 62 62 96 6D 2F 13 49 C9 86 91 C5 22 DC E3 08 43
```

The first four bytes are `NXS\0`. The remaining 16 bytes are the raw bytes of the format's UUIDv4 signature (`6262966d-2f13-49c9-8691-c522dce30843`). Treat the entire 20-byte sequence as the magic value; do not byte-swap or parse it as an integer.

The UUID suffix has 122 random bits, making accidental collision very unlikely. This does not prove global uniqueness, imply registration, or prevent another format from choosing the same bytes.

## Fixed header

The fixed header is 80 bytes, encoded using `<20sHHIIQQ32s`:

| Offset | Size | Field | Value / meaning |
|---:|---:|---|---|
| 0 | 20 | `magic` | Exact signature above |
| 20 | 2 | `major` | Unsigned little-endian integer; `0` |
| 22 | 2 | `minor` | Unsigned little-endian integer; `2` |
| 24 | 4 | `header_size` | Unsigned little-endian integer; `80` |
| 28 | 4 | `flags` | Must be `0` in version 0.2 |
| 32 | 8 | `manifest_offset` | Absolute byte offset of the manifest |
| 40 | 8 | `manifest_length` | Manifest length in bytes |
| 48 | 32 | `manifest_sha256` | SHA-256 digest of the exact manifest bytes |

## Archive layout

1. The 80-byte fixed header.
2. Internal payload chunks, each beginning at a 4096-byte-aligned absolute file offset. Alignment padding is zero-filled.
3. A UTF-8 JSON manifest at `manifest_offset`; it ends at EOF.

The header points to the manifest so a remote reader can fetch the header, then the manifest, then only the payload ranges it needs using range reads. The payload is not required to be loaded into RAM as a whole.

Each manifest entry has a unique `name`, a `kind` (for example, `tensor` or `asset`), `nbytes`, a SHA-256 digest of the complete entry, and a list of internal chunks. Tensor entries also carry the source `dtype` and `shape`. Each chunk records its absolute `offset`, `length`, `logical_offset` within the entry, and SHA-256 digest. Chunk order and logical ranges must be contiguous; payload ranges must not overlap the header, one another, or the manifest.

Illustrative manifest shape:

```json
{
  "format": "DOTNXS",
  "version": [0, 2],
  "source": {"format": "safetensors"},
  "archive_metadata": {},
  "entries": [
    {
      "name": "model.layers.0.weight",
      "kind": "tensor",
      "dtype": "BF16",
      "shape": [1024, 1024],
      "nbytes": 2097152,
      "sha256": "...",
      "chunks": [
        {"offset": 4096, "length": 2097152, "logical_offset": 0, "sha256": "..."}
      ]
    }
  ]
}
```

Chunk size is a writer/runtime policy, not a format constant. Chunking must preserve the original entry byte sequence exactly. Version 0.2 does not define payload compression or quantization.

## Validation and compatibility

- Readers must validate the full magic, version, header size, zero flags, manifest bounds, and manifest digest before using the index.
- Every entry's chunk lengths must sum to `nbytes`; each chunk and complete entry digest can be checked while reading.
- A reader must reject overlapping/out-of-bounds ranges, duplicate names, invalid JSON, or unsupported versions.
- Version 0.2 requires `major = 0`, `minor = 2`, `header_size = 80`, and `flags = 0`.

## Converter scope

The converters copy tensor payload bytes or Parquet shard bytes into internal chunks and record source names, dtypes/shapes when applicable, and metadata. They are streaming and lossless. `parquet2nxs` records Arrow schema and row-group summaries but does not define a Parquet query engine, embeddings, or RAG retrieval semantics. Transformers/Hugging Face wrappers and model execution are separate layers; converter success alone does not imply that a model can run.
