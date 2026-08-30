# Browser media fixtures

These deterministic fixtures cover browser-side audio and video handling without network access:

- `tone.mp4`, `tone.m4a`, `tone.mp3`, and `tone.wav`: four-second 220 Hz samples.
- `tone-30.wav`: 30-second 220 Hz sample used by long recording and sharing checks.
- `tone-hevc.mov`: four-second HEVC/H.265 video with an AAC 220 Hz audio track.
- `tone-long.mp4`: 181-second H.264 video with an AAC 220 Hz audio track.
- `no-audio.mp4`: four-second H.264 video without an audio track.

`tests/helpers/generate-media-fixtures.mjs` copies the reusable files into the ignored test workspace and creates the large, corrupt, and empty boundary fixtures.
