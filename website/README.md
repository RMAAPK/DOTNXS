# DOTNXS website

This is a React Native Web frontend built with Vite. Render publishes `website/dist` as the static site for `https://dotnxs.rmaa.pk`; the browser sends chat requests to the separate C++ API service and runs no model weights locally.

- `npm install` installs the frontend dependencies.
- `npm run dev` starts a local Vite server.
- `npm run build` creates `dist/` for static hosting.

The Render API hostname is set in `public/config.js`. The companion `public/ping.html` page calls the C++ service's `/ping` endpoint and can poll every ten minutes while the page stays open.
