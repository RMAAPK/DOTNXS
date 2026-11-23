# DOTNXS (`.nxs`)

DOTNXS is a single-file archive format for AI/LLM artifacts. One `.nxs` file contains internal tensor and asset chunks, with a manifest for selective reads and bounded caching.

The CLI wraps a C++17/pybind11 archive writer, while Python handles conversion and the experimental streamed Inkling runtime. The runtime reads the original `Inkling-2-Mini.nxs` directly, pages dense tensors and selected NVFP4/MXFP8 experts on demand, and keeps the NXS chunk cache capped at 200 MiB. GitHub Actions builds and tests the native wheel on Linux, macOS, and Windows.

- [Format specification](FORMAT.md)
- [Python package and CLI](pyproject.toml)
- License: MIT (see [LICENSE](LICENSE))

## Install

```bash
python -m pip install --upgrade dotnxs
```

On Windows, `pip` can install console scripts into the per-user `Scripts` directory while that directory is absent from `PATH`. If PowerShell says `dotnxs` is not recognized, the package is still installed; run `python -m dotnxs.cli --help` or add the `Scripts` directory shown in pip's warning to `PATH`. The source package also supports `python -m dotnxs`.

## CLI

```bash
dotnxs --help
dotnxs convert safetensors input.safetensors output.nxs
dotnxs inspect output.nxs
dotnxs verify output.nxs
```

The default internal chunk size is 4 MiB and can be changed with `--chunk-mib`. Conversion preserves source tensor bytes; it does not quantize weights.

Convert a local Transformers directory (Safetensors weights plus config/tokenizer assets):

```bash
dotnxs convert transformers ./model-directory model.nxs
```

Convert a Hugging Face model repo (requires the optional Hub client):

```bash
python -m pip install 'dotnxs[hf]'
dotnxs convert hf thinkingmachines/Inkling-2-Mini inkling.nxs
```

`hf2nxs` reads the Hub file index and fetches Safetensors tensor ranges over HTTP into one `.nxs` archive, avoiding a full local checkpoint copy. Range requests must be supported by the serving endpoint.

## Experimental Inkling runtime and Render demo

Install the CPU runtime dependencies with `python -m pip install '.[runtime]'`. The experimental text runner uses Transformers' Inkling architecture and fetches the checkpoint from `rmaaorg/Inkling-2-Mini-NXS`; it does not stage the whole model locally. The Render demo source is split into:

- `website/`: React Native Web frontend built with Vite for `dotnxs.rmaa.pk`.
- `server/`: C++ HTTP gateway and Dockerfile, plus a private Python worker that runs the official Transformers model math.

Create a Render Static Site with Root Directory `website` and Publish Directory `build`, plus a Docker Web Service with Root Directory `server`, Dockerfile `./Dockerfile`, and Docker Context `.`. The C++ gateway answers `/ping` with `pong` and streams chat responses from the local model worker. The Render API hostname is set in `website/public/config.js`.

Render's Free web plan has 512 MB RAM; that may be insufficient for the Torch worker and its temporary weights even with a 200 MiB NXS chunk-cache cap. The Free service also sleeps after 15 minutes without inbound requests. `website/public/ping.html` offers a ten-minute heartbeat while open, and `.github/workflows/render-ping.yml` can ping `/ping` on a schedule. Scheduled pings consume the workspace's Free instance hours; they do not bypass Render's monthly limits.

The server may use a paid Render plan for reliable RAM. The 200 MiB value bounds verified archive chunks, not total process memory. Inkling executes 48 layer applications per generated token, so CPU responses can be slow.

Validation so far: tokenizer initialization and one real decoder layer have run using the public NXS archive. A GitHub Actions workflow is configured to compile and smoke-test the C++ gateway; the Docker image and full 48-application chat generation have not yet completed end-to-end validation.

Run the converter and runtime unit tests with:

```bash
python -m unittest discover -s tests -v
```

Most unit tests use tiny generated fixtures. The one-layer live check is not an end-to-end model-generation test.

The format's 20-byte signature is fixed. Its 16-byte suffix is a UUIDv4-derived value for a very low accidental-collision probability, not a claim of globally guaranteed uniqueness.
