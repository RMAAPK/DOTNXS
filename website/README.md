# DOTNXS website

This is a React Native Web frontend built with Vite. Render publishes `website/build` as the static site for `https://dotnxs.rmaa.pk`; the browser loads the public Hugging Face tokenizer files from the exact model revision reported by the API, renders and tokenizes prompts locally, then sends token IDs to the separate C++ API service. Model weights are never loaded in the browser.

- `npm install` installs the frontend dependencies.
- `npm run dev` starts a local Vite server.
- `npm run build` creates `build/`, matching the existing Render Static Site Publish Directory.
- `npm test` runs the browser-tokenizer prompt-template and bounded-context tests.

The Render API hostname is set in `public/config.js`. The companion `public/ping.html` page calls the C++ service's `/ping` endpoint and can poll every ten minutes while the page stays open.
