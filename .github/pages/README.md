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

The current October mixer demo is a 20-second, 30 fps recording of the actual
control page with two live WebRTC viewers and a visible upright T-bar. Operate
the page while recording its display; do not substitute rendered state or recorded
pictures for the live viewers. Check both viewers are decoding before capture.
The tools are in [tools/mixer-recording](../../tools/mixer-recording/README.md).
Publish sanitized capture metadata in `capture-20261028.json`; keep historical
capacity and latency measurements unchanged. Earlier host-output recordings and
September graph images remain historical release assets.
