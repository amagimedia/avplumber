# Demo review

Reviewed 2026-09-07 from a fresh public checkout on an NVIDIA T4. All five demos
built using public sources. No custom TensorRT archive, private image,
company account, or private media was needed for the builds and tests.

## Which demo to choose

**Playlist** is the easiest starting point: generated clips, a terminal UI,
and a GPU-free UI preview. **Mixer** demonstrates layouts and transitions well.
**Replay** demonstrates frame/time seeking, scrubbing, reverse, and speed controls.
Its paused-seek and EOF recovery bugs are fixed and covered by native tests. **Browser capture** needs the most host graphics setup; its bundled
animation checks capture, and users can supply their own page for a presentation.
**CUDA overlay** provides a deterministic correctness check with inspectable images.

## Findings addressed

- Added the main-page comparison and shared Docker/NVIDIA setup guide.
- Documented the public mixer image as the base for Playlist and Replay.
- Replaced the private Electron mirror with official releases and the bundled
  NVIDIA GBM shim; used the bundled animation instead of hosted graphics.
- Fixed the browser consumer's CUDA target, which previously failed to compile
  the bundled `overlay_many_cuda` filter.
- Mounted the browser graph directory with its helpers and SELinux labeling;
  mounting only its entry script caused `ModuleNotFoundError`.
- Selected the NVIDIA runtime for graphics services and searched both Ubuntu
  and Fedora GBM directories. Removed an incorrect hardcoded GBM backend link.
- Made the single-browser graph wait for a captured CUDA frame before starting
  its encoder/output. Starting output first failed with `timeBase(): out ctx none`.
- Clarified that these five media demos do not need TensorRT or neural models.

## Verification

| Demo | Completed checks | Result |
| --- | --- | --- |
| Mixer | 24 local tests; Docker build; all layouts, Cut, Fade, four wipes; continued encoding after source-reader loss; decoded output inspection. | Passed. |
| Playlist | 130 local tests; Docker build and generated clips; 94 live regression checks; advancing 1080p WebRTC video. | Passed. |
| Replay | 66 local tests; NVIDIA and CPU builds; 155 distinct native integration cases per backend; regular-input all-intra and B-frame transcodes; 2,160-frame conversion and WebRTC playback. | Passed. |
| Browser capture | Five local tests; full Docker stack; advancing WebRTC video from one 720p60 page and the default eight-source 1080p60 grid. | Passed after setup/startup fixes. |
| CUDA overlay | Five local tests; dedicated Docker build; all 45 GPU cases compared against the CPU reference. | Passed with zero failed cases. |

All 230 local tests passed across the five demo suites. Local suites were
run separately to avoid test-module name collisions; Textual 8.2.8 enabled the
TUI tests. Native tests used generated media. The overlay report records patch
hashes and per-plane comparisons. Video screenshots were inspected as well as
checking that decoded frames advanced. Separate checks confirmed exact one-frame
steps in both directions in Replay.

These checks establish functional coverage on the tested configuration.
The browser demo's published performance figures use a different page; benchmark
the intended content before making capacity claims.

## Replay fixes and regression coverage

The paused-seek bug was reproduced with a 72-second, 30-fps recording: seeking
from 6,000 ms to **35,983 ms** stalled, while **36,000 ms** worked. The seek table
selected frame 1079 at 35,966.667 ms (indexed as 35,967 ms), but the decoder's
cutoff was based on the requested time minus 7 ms: 35,976 ms. It discarded the
only frame sent by the paused reader. Indexed seeks now resolve the selected
frame before configuring the decoder cutoff.

The expanded suite also exposed repeated frames when changing active playback
from 50% to 100%, and inability to seek after non-looping EOF. Replay now drains
old cadence before changes to speeds at or below 100%. Its reader and demux stay
available at EOF, and its decoder drains the final frames while remaining
reusable. The regular `input` source is unchanged; decoder/demux EOF changes are
opt-in, and ordinary finite pipelines retain their defaults.

The same suite runs on NVDEC/NVENC with FFmpeg 7.1.5 and on software H.264/libx264
with FFmpeg 7.1.4. The CPU module was built with CUDA disabled and tested using
`runc`, without NVIDIA devices or CUDA linkage. Each backend passed the original
119-test full run and 44 additional slow-speed checks (eight extend existing
cases, giving **155 distinct cases**). Coverage includes:

- 24/25/30/60 fps; 25/50/100/200% speeds in both directions.
- Raw absolute, relative, and UTC seeks around frame boundaries, nearest-frame
  ties, and the old 7-ms cutoff, approached from both earlier and later positions.
- Fresh exact-frame observations, stable paused holds, exact frame steps,
  repeated/seeded seeks, bursts, and clamping at both recording boundaries.
- Repeated active speed transitions, looping, final-frame draining, repeated
  seeks during EOF draining, and resuming after EOF.
- Regular `input` with both all-intra and B-frame H.264 sources: every input frame
  must reach the output and seek table. Configured RTP headers are also checked.

The built-in exercise was tightened too: its old one-frame tolerance could mark
an incorrect nudge as passing. It now requires fresh exact seek/nudge results,
a stable paused observation, and the final target after rapid seeks. Fault tests
verify that stale and off-by-one nudge results are rejected.

After tightening the checker, all four frame-rate exercise cases were rerun and
passed on each backend. The original 72-second recording also passed all **25
built-in checks on both backends**, with no skips. Seeking to 35,983 ms now yields
frame 1079 at 35,967 ms, and the one-frame nudges report their exact target frames.

## Recorded Replay demo

The [49-second MP4](https://github.com/amagimedia/avplumber/releases/download/replay-demo-media-2026-09/replay-demo.mp4) shows the real TUI
beside its Janus/WebRTC output, using generated media with a source frame/time
counter. It covers paused steps, media/UTC seeks, 0.25×–2× speed, scrubbing,
reverse, EOF, and seeking afterward. H.264, 1920×1080, 15 fps, 1.7 MB; the full
file decoded without errors. Paused seek and EOF frames were visually checked
against the TUI's source-frame counter.

The [web UI screenshot](https://github.com/amagimedia/avplumber/releases/download/replay-demo-media-2026-09/replay-graph.png) includes all
14 nodes and 13 queues, with live occupancy and throughput statistics. The
[published demo page](https://amagimedia.github.io/avplumber/demos/replay/docs/) includes chapter buttons
and the full-resolution graph. GitHub Pages embeds public release assets; the
new MP4, JPEG, and PNG were not committed to Git. Chrome playback and chapter
seeking were checked against the live page.

### Backward-seek recheck

After publication, 124 seek/pause/EOF cases were rerun on each backend:
software H.264/libx264 passed in 162.52 seconds, and NVDEC/NVENC passed in
349.72 seconds. Neither run had failures or skips; 24 other playback cases
were deliberately deselected. The NVIDIA fixture generator used the distro
FFmpeg with libx264; the native application retained its custom FFmpeg 7.1.5.

The published recording was also inspected directly: at approximately 6.9 s,
the -1-frame step shows 402 → 401; at 9.4 s, the -30-frame step shows 431 → 401.
The backward UTC seek shows frame 300 at 10.000 s, and the backward seek after
EOF shows frame 1079 at 35.967 s. Video and TUI agree at these paused positions.
During reverse playback, the source frames decrease; the slower TUI refresh
can briefly trail the video counter.
