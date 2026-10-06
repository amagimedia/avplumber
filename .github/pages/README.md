# Demo publishing

GitHub Pages publishes the `gh-pages` branch. Edit mixer cookbook HTML, CSS,
screenshots and captured reports there; the code branch links to the
[live cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/).

Large recordings and graph images are release assets:

- [Replay media](https://github.com/amagimedia/avplumber/releases/tag/replay-demo-media-2026-09)
- [DMA-BUF media](https://github.com/amagimedia/avplumber/releases/tag/dmabuf-demo-media-2026-09)

Upload replacement media to a release, then update its URLs and SHA-256 manifest
on `gh-pages`. Do not add recording binaries to either branch.

When refreshing the cookbook, review the maintained mixer Markdown in the code
branch too. Pin implementation links to the reviewed code revision, repair
moved paths and stale line anchors, and distinguish implementation status,
GPU correctness tests, capacity measurements and actual deployment state.
Keep historical measurements attached to their workload and revision; a code
review or small smoke test does not refresh a capacity benchmark. Publish
sanitized textual evidence without private host or storage paths.
