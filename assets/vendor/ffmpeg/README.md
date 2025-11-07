# ffmpeg.wasm vendored assets

The ffmpeg fallback mirrors the upstream worker and core assets so they can be served from the same origin as the app:

- `worker.js` from `@ffmpeg/ffmpeg@0.12.15/dist/esm/worker.js`
- `ffmpeg-core.js` from `@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js`
- `ffmpeg-core.wasm` from `@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm`

These files are downloaded automatically in CI/CD via `npm run fetch:ffmpeg`, which pulls the exact versions above into this directory before running tests or publishing to GitHub Pages. For local development you can trigger the same download manually:

```
npm run fetch:ffmpeg
```

If you need to audit or mirror the assets yourself, fetch them from the upstream packages and drop them in this folder so `tests/verify-ffmpeg-download.mjs` can verify their presence. The `.wasm` binary is optional for contributors because GitHub's file size limit prevents checking it in; the test suite only enforces the presence of the JavaScript shims and treats the WebAssembly file as a best-effort download. At runtime the app will fall back to streaming the assets directly from the CDN when local copies are missing, but shipping the vendored versions keeps worker loading reliable across browsers and improves startup time.
