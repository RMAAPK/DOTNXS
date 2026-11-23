---
title: DOTNXS Inkling Chat
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Inkling chat shell with a bounded NXS cache
---

# DOTNXS Inkling Chat

This static Space contains the chat interface and a fixed 200 MiB lazy-cache policy for the real [`Inkling-2-Mini.nxs`](https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS) archive. A collapsed developer diagnostic can fetch its header, manifest, and tensor ranges over HTTP and verify SHA-256 without downloading the full archive.

**Model inference is not connected yet.** The chat input stays disabled until a real CPU inference runtime is available; the Space does not generate placeholder replies.

The Space UI is MIT-licensed. The archived Inkling 2 Mini weights and upstream materials remain under their original **Apache-2.0** license and applicable model terms; see the model repository's `LICENSE` and README.
