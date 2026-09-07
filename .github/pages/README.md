# Demo media hosting

GitHub Pages publishes the existing `gh-pages` branch. The Replay HTML page
embeds the MP4, first-frame JPEG, and graph PNG from the
[Replay demo media release](https://github.com/amagimedia/avplumber/releases/tag/replay-demo-media-2026-09).
The media files are release assets, not Git blobs.

To replace a recording, upload the media to a new release, update the URLs in
the Replay README and HTML page, and update the SHA-256 manifest. Copy the HTML
page to the same location on `gh-pages`, preserving the other demo files, then
push both branches. Never add the media files to either branch.
