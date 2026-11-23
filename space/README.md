---
title: DOTNXS Inkling Chat
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Experimental CPU chat from streamed NXS weights
---

# DOTNXS · Inkling 2 Mini Chat

This experimental Space runs text-only Inkling 2 Mini inference from the original single [`Inkling-2-Mini.nxs`](https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS) archive. The CPU runner uses Transformers for the model's attention and loop math, while loading dense layer tensors on demand and fetching only router-selected NVFP4/MXFP8 experts. The output vocabulary matrix is streamed in row blocks rather than loaded whole.

The remote NXS chunk cache is fixed at **200 MiB**, lazy, and LRU-evicted; it is not allocated at startup. The demo is CPU-only, text-only, and deliberately limits context and output length. Inkling executes 48 layer applications per token, so response latency may be high. This is real model inference, not a canned response or range-test page.

The Space UI is MIT-licensed. The archived Inkling 2 Mini weights and upstream materials remain under their original **Apache-2.0** license and applicable model terms; see the model repository's `LICENSE` and README.
