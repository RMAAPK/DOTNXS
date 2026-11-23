# DOTNXS website

Render publishes this directory as the static site for `https://dotnxs.rmaa.pk`. The page sends text chat requests to the FastAPI server configured in `config.js`; it contains no model weights and runs no inference in the browser.

If Render assigns a different API hostname, change `window.DOTNXS_CONFIG.apiBaseUrl` in `config.js` before redeploying the static site. The server's CORS allowlist is configured by `CORS_ORIGINS` in the root `render.yaml`.
