---
title: DOTNXS Stream Lab
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Browser range and WebGPU probe for NXS archives
---

# DOTNXS Stream Lab

This static Space probes the real [`Inkling-2-Mini.nxs`](https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS) archive using HTTP byte ranges. It fetches the fixed header, manifest, and one internal tensor chunk, then verifies the manifest and chunk SHA-256 digests. It does **not** download the whole archive or run model inference yet.

The Space's UI code is MIT-licensed. The archived Inkling 2 Mini weights and upstream materials remain under their original **Apache-2.0** license and applicable model terms; see the model repository's `LICENSE` and README.

The WebGPU status display is a device-capability check only. A model-specific WebGPU inference runtime is a future step.
