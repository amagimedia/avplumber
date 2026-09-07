# Demo media hosting

GitHub Pages is deployed by the **Publish demo pages** Actions workflow.
It preserves the existing `gh-pages` content, adds the Replay HTML page, and
fetches the MP4, first-frame JPEG, and graph PNG from the
[Replay demo media release](https://github.com/amagimedia/avplumber/releases/tag/replay-demo-media-2026-09).
The new media files are release assets and deployment artifacts, not Git blobs.

To replace a recording, upload the media to a new release, update the release tag
in the workflow and the SHA-256 manifest in the Replay docs directory, then push
those text changes. The workflow checks the downloaded files before deploying.
Run **Publish demo pages** manually after updating older content on `gh-pages`.
