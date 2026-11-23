---
title: DOTNXS Stream Lab
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: CPU-side NXS range streaming with a 200 MiB cache
---

# DOTNXS Stream Lab

This static Space probes the real [`Inkling-2-Mini.nxs`](https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS) archive using CPU-side HTTP byte ranges and a lazy 200 MiB in-memory cache. It fetches the fixed header, manifest, and one internal tensor chunk, then verifies the manifest and chunk SHA-256 digests. It does **not** download the whole archive or run model inference yet.

The Space's UI code is MIT-licensed. The archived Inkling 2 Mini weights and upstream materials remain under their original **Apache-2.0** license and applicable model terms; see the model repository's `LICENSE` and README.

A model-specific CPU inference runtime, expert routing, and KV-cache handling are future work; the page does not generate placeholder chat replies.
