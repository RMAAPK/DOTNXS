# DOTNXS Render server

`cpp_server.cpp` is the public C++ HTTP gateway. It provides `/ping` (`pong`) and `/health`, and forwards `/v1/chat` to the private Python worker in `worker.py`. The Python worker retains the official Transformers model math; the gateway does not pretend to be a C++ Inkling inference engine.

## Manual Render setup

1. Create a **Static Site** from `RMAAPK/DOTNXS`, Root Directory `website`, build command `npm ci --no-audit --no-fund && npm run build`, and Publish Directory `dist` (Vite does not produce a `build/` folder). Add custom domain `dotnxs.rmaa.pk` and configure the DNS record Render provides.
2. Create a **Web Service** from the same repository, runtime **Docker**, Dockerfile path `server/Dockerfile`, Docker context `.` (the repository root). Leave the Web Service Root Directory blank. **Do not** set either context or Root Directory to `server`: the Docker build needs the root `pyproject.toml` and `dotnxs/` package. The image starts the C++ gateway and its local Python inference worker.
3. The website uses `https://dotnxs-inkling-api.onrender.com` from `website/public/config.js`; if Render assigns a different service URL, update it and redeploy the static site.
4. Set the service environment variables if needed: `NXS_REPO_ID`, `NXS_FILENAME`, `DOTNXS_CACHE_BYTES`, `DOTNXS_MAX_CONTEXT_TOKENS`, `DOTNXS_MAX_NEW_TOKENS`, and `CORS_ORIGINS`. Defaults are set for the public NXS repo and `dotnxs.rmaa.pk`.

The C++ service has `/ping`, which returns `pong` without initializing the model. Render Free sleeps after 15 idle minutes. `website/public/ping.html` can ping every ten minutes while open; for a background heartbeat, set the GitHub repository variable `RENDER_PING_URL` to `https://dotnxs-inkling-api.onrender.com/ping`. The scheduled GitHub Actions workflow calls it every ten minutes. This avoids ten separate Render Cron services, each of which has a minimum charge.

## Free-plan limits

Render's Free web service has 512 MB RAM and spins down after 15 minutes without inbound traffic. The heartbeat can keep it awake, but it consumes the workspace's monthly Free instance hours (750 hours total) and does not bypass that limit. The Torch worker plus active weights may exceed 512 MB, and Render may suspend free services for unusually high outbound traffic. A real decoder-layer check fetched about 234 MB; full token generation can require several GB of streamed weights. Ping does not remove those RAM or bandwidth limits. Render's 2 CPU / 4 GB `2c-4g` plan is paid.

The API serializes generation, caps prompts at 64 tokens and replies at 12 tokens, and is text-only. Inkling executes 48 layer applications per token, so CPU response latency can be high. The public NXS repo does not require `HF_TOKEN`; if you add one for a private repo, set it as a Render secret, not in source.
