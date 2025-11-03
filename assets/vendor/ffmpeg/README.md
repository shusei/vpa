# ffmpeg.wasm vendored assets

The ffmpeg fallback requires the upstream worker and core assets to be served from the same origin as the app:

- `worker.js` from `@ffmpeg/ffmpeg@0.12.15/dist/esm/worker.js`
- `ffmpeg-core.js` from `@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js`
- `ffmpeg-core.wasm` from `@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm`

These files are downloaded automatically in CI/CD via `npm run fetch:ffmpeg`, which pulls the exact versions above into this directory before running tests or publishing to GitHub Pages. For local development you can trigger the same download manually:

```
npm run fetch:ffmpeg
```

If you need to audit or mirror the assets yourself, fetch them from the upstream packages and drop them in this folder so `tests/verify-ffmpeg-download.mjs` can verify their presence.
