# Demo publishing

GitHub Pages publishes the `gh-pages` branch. Edit mixer cookbook HTML, CSS,
screenshots and captured reports there; the code branch links to the
[live cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/).

Large recordings and graph images are release assets:

- [Current mixer media](https://github.com/amagimedia/avplumber/releases/tag/mixer-demo-media-2026-10)
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

The current October mixer demo replays 20 seconds in the actual control page at
2× Retina scale, with two viewers, an upright T-bar and sampled performance meters.
Capture both encoded outputs on the host before Janus and record the real UI's
control trace and state concurrently. Render offline at 30 fps; validate native
packet continuity and ticker motion before publishing. The tools are in
[tools/mixer-recording](../../tools/mixer-recording/README.md).
Publish sanitized metadata in `capture-20261028.json`, clearly identifying the UI
and metrics as replayed. Keep historical capacity and latency measurements unchanged.
Earlier recordings remain historical release assets.
