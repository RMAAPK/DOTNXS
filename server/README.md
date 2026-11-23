# DOTNXS Render server

This directory contains the Render-facing FastAPI service. The frontend lives in `website/`; both services are declared in the repository-root `render.yaml`.

## Deploy from GitHub

1. In Render, create a **Blueprint** and connect `RMAAPK/DOTNXS` on the `main` branch.
2. Review `render.yaml`. It defines `dotnxs-inkling-api` (FastAPI) and `dotnxs-website` (static site).
3. The API uses Render plan `2c-4g` (2 CPUs, 4 GB RAM). This is a paid compute plan; the free 512 MB plan is not suitable for PyTorch plus the streamed model runtime. Do not sync the Blueprint unless that plan and its cost are acceptable.
4. The static site requests `dotnxs.rmaa.pk`. Add/verify the DNS record Render shows for that domain.
5. The website's `website/config.js` points to `https://dotnxs-inkling-api.onrender.com`. If Render assigns a different service hostname, update that file and redeploy.

Render installs the DOTNXS `runtime` extra and starts `uvicorn server.app:app` on `$PORT`. `server/requirements.txt` pins the CPU-only PyTorch wheel. The service exposes `GET /health` and `POST /v1/chat` (server-sent events). It reads the public NXS archive directly from Hugging Face; it does not clone or download the full model checkpoint. The server's verified NXS chunk LRU cache is fixed at 200 MiB.

## Limits and latency

The API serializes generation, caps prompts at 64 tokens and replies at 12 tokens, and is text-only. Inkling 2 Mini executes 48 layer applications per token. NXS streaming keeps the chunk cache bounded but does not make the model lightweight; first responses can be slow. The Render instance also needs enough RAM for PyTorch, the current layer, selected experts, activations, and KV state in addition to the 200 MiB archive cache.

`HF_TOKEN` is optional because the archive repo is public. If you add one, configure it as a secret in Render, never in `render.yaml` or the static website.
