# Demo media hosting

GitHub Pages publishes the existing `gh-pages` branch. Demo HTML pages embed
MP4 recordings, JPEG posters, and web UI graph PNGs from releases:

- [Replay demo media](https://github.com/amagimedia/avplumber/releases/tag/replay-demo-media-2026-09)
- [DMA-BUF browser demo media](https://github.com/amagimedia/avplumber/releases/tag/dmabuf-demo-media-2026-09)

The media files are release assets, not Git blobs.

To replace a recording, upload the media to a new release, update the URLs in
the demo README and HTML page, and update its SHA-256 manifest. Copy the HTML
page and manifest to the same location on `gh-pages`, preserving the other demo
files, then push both branches. Never add the media files to either branch.
