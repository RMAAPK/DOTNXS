---
title: DOTNXS Inkling Chat
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Inkling NXS CPU inference prototype
---

# DOTNXS Inkling CPU Prototype

This static Space is an honest status page for the DOTNXS-backed Inkling runtime. The Python prototype can fetch and execute one routed MoE layer using the original NVFP4/MXFP8 expert tensors from the single [`Inkling-2-Mini.nxs`](https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS) archive.

**Full chat generation is not connected yet.** Attention, loop/message state, embeddings, tokenizer integration, and autoregressive decoding remain unfinished, so the chat controls stay disabled and no placeholder replies are produced. The former browser-side archive/range test has been removed from the demo.

The Space UI is MIT-licensed. The archived Inkling 2 Mini weights and upstream materials remain under their original **Apache-2.0** license and applicable model terms; see the model repository's `LICENSE` and README.
