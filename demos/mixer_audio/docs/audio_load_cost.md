# What the audio mixer costs — `mixer_audio` vs `mixer`, 96 sources

Measured on the AWS Fedora demo host (`avplumber-lamus-2`, 8 vCPU, 30 GiB RAM,
NVIDIA Tesla T4 16 GiB, driver 615.71.09) on 2026-10-26, image
`avplumber-mixer:local` built 14:47 UTC from `12b52a9`.

Both runs used the **same** `/media/demo.json`, so the source configuration is
identical: 96 sources (72 h264 1080p files + 24 v210 raw), 24 scenes, 1920×1080
HLG `p210le` canvas at 30 fps, two program renditions (SDR h264_nvenc `p3`
tonemapped, HDR hevc_nvenc `main10` `p3`), no browser windows, no DSK.

| Run | Demo | Audio | Window |
|---|---|---|---|
| A | `docker-compose/mixer_audio/compose.yaml` | 72 AAC tracks → `audio_mixer` → Opus/RTP | 150 s |
| B1, B2 | `docker-compose/mixer/compose.yaml` | none | 150 s each |

Sampling used [`scripts/sample_load.py`](../scripts/sample_load.py) (the host has
no sysstat): `/proc/stat` and cgroup v2 deltas over the whole window, 1 Hz
`nvidia-smi`, and per-thread `/proc/<pid>/task/*/stat` deltas. B1/B2 agree within
1 % of each other, so one audio sample is enough to read the difference.

## Result

Audio costs **~0.44 CPU cores of the mixer process (+12.5 %)** and **~0.58 cores
host-wide (+7.3 pp of 8 vCPU)**. It costs **no measurable GPU**.

| Metric | A: with audio | B: no audio (mean) | Δ |
|---|---:|---:|---:|
| Mixer process CPU | 4.00 cores | 3.55 cores | **+0.44 cores (+12.5 %)** |
| — user / system | 1.99 / 2.01 | 1.68 / 1.88 | +0.31 / +0.14 |
| Mixer container CPU | 4.11 cores | 3.61 cores | +0.49 cores |
| `janus` container CPU | 0.057 cores | 0.041 cores | +0.016 cores |
| Host CPU busy | 62.95 % | 55.65 % | **+7.3 pp (= +0.58 cores)** |
| — user / system / irq | 26.5 / 27.3 / 8.8 % | 22.1 / 24.9 / 8.3 % | +4.4 / +2.4 / +0.5 pp |
| Load average (1 min, mean) | 17.7 | 14.7 | +3.0 |
| Mixer threads | 678 | 520 | **+158** |
| Mixer RSS | 7428 MB | 7285 MB | +143 MB |
| GPU SM util | 81.2 % | 81.3 % | −0.1 pp |
| GPU NVDEC util | 99.5 % | 100 % | −0.5 pp |
| GPU NVENC util | 16.1 % | 16.4 % | −0.3 pp |
| GPU power | 67.9 W | 67.7 W | +0.2 W |
| GPU memory | 9515 MiB | 9204 MiB | +311 MiB (not audio — see caveats) |

Neither configuration dropped a frame: both delivered exactly 4500 program
frames per 150 s window (30.00 fps) with `missed_deadlines == 0`. The audio chain
buys its cost out of headroom, it does not cost frames — at 96 sources this host
is bound by NVDEC (~100 % in both runs), not by the audio mixer.

## Where the 0.44 cores go

Per-thread deltas, in % of one core. avplumber names its worker threads after
graph nodes, so the chain can be attributed without a profiler. Thread names are
truncated to 15 characters by the kernel.

| Thread group | A: with audio | B: no audio | Δ | What it is |
|---|---:|---:|---:|---|
| `decode_audio_N` (72) | 11.77 | — | **+11.77** | per-source AAC decode |
| `resample_audio_N` (72) | 9.29 | — | **+9.29** | per-source resample to 48 kHz stereo |
| `audio_mixer` (1) | 8.92 | — | **+8.92** | the AFV mixing node itself, 72 → stereo |
| `EventLoop` (5) | 12.85 | 7.66 | +5.19 | source pacing, now also A/V sync teams |
| `decode_N` (96) | 97.77 | 94.24 | +3.53 | video decode, same work either way |
| `input_N` (96) | 123.99 | 120.61 | +3.38 | demux: one node feeds video **and** audio packets |
| `janus_audio_enc` + `janus_audio_lim` | 1.66 | — | +1.66 | Opus encode and the downstream limiter |
| everything else | — | — | +0.6 | colour/OTM/compositor/udp-tx noise |
| **Sum of named deltas** | | | **+43.7** | vs **+44.4** measured on the process |

So roughly two thirds of the cost is the per-source chain (decode + resample +
the demux/sync share, ~0.28 cores for 72 sources ≈ **3.9 mcores per source**) and
one third is the fixed mixing/encode tail (`audio_mixer` + Opus ≈ **0.11 cores**,
independent of source count).

`audio_mixer` is a single thread at 8.9 % of a core while summing 72 live stereo
inputs — it is nowhere near a bottleneck, but it is also single-threaded, so it
would be the first audio component to cap out if source counts grew by an order
of magnitude.

## Caveats

1. **The two demo trees have diverged beyond audio.** The video-only demo's
   `prepare_demo.py` also declares `decoded_format` (`nv12` ×48, `p010le` ×24) on
   its 72 file sources, which `mixer_audio`'s does not. CPU-wise this changed
   nothing measurable (the SDR colour stage ran in both runs: 60 threads,
   7.55 vs 7.21 % of a core), but it is the likeliest explanation for the
   **+311 MiB GPU memory** in run A — do not read that row as an audio cost. The
   CPU figures above are a fair audio attribution; the GPU memory figure is not.
2. **Assets differ only by the audio track.** `mixer_audio` renders
   `synthetic_v6_*` (same generator, same `encoding_options`: `-bf 0
   -dpb_size 2`, same per-source bitrate ladder) plus a per-source sine as AAC
   stereo 48 kHz; `mixer` renders `synthetic_v5_*` without it. Video encode
   parameters are identical, so decode cost is comparable (`decode_N` differs by
   3.5 % of a core, i.e. the audio packets travelling with the video).
3. **Program slot asymmetry.** Run A had PGM in slot B after earlier cuts (slot A
   idle), runs B1/B2 drew both compositor slots from a fresh start. The
   compositor CPU difference is small and of mixed sign (`mixer_comp_a` −0.60,
   `mixer_comp_b` +0.08 % of a core) and GPU SM util was identical, so this does
   not move the conclusion.
4. **One audio sample, two video samples.** The planned A-B-A repeat was not
   taken: the audio demo would not restart (see below).
5. The known cosmetic `[resample_audio_N] output PTSes too large, dropping
   output, 256 samples` line fires ~23×/s at 96 sources (13716 lines in 10 min)
   and its logging is inside the measured cost.

## Reproducing it

```sh
scp -i ~/.ssh/awsdev.pem demos/mixer_audio/scripts/sample_load.py fedora@<HOST>:~/loadcmp/
ssh -i ~/.ssh/awsdev.pem fedora@<HOST>
cd /home/fedora/audio-video-mixer/avplumber
# A: audio (default stack)
python3 -I ~/loadcmp/sample_load.py --seconds 150 --label audio > ~/loadcmp/audio.json
# B: same /media/demo.json, no audio. Same compose project name, so this swaps
# the mixer entrypoint/janus env in place; nothing else has to change.
docker compose -f docker-compose/mixer/compose.yaml up -d      # first run regenerates synthetic_v5_* (~145 s)
python3 -I ~/loadcmp/sample_load.py --seconds 150 --label video > ~/loadcmp/video.json
```

Raw reports stayed on the host in `/home/fedora/loadcmp/` (`audio_a1.json`,
`video_b1.json`, `video_b2.json`) together with the generated
`graph_audio.json` / `graph_video.json` recipes.

### Two traps when swapping to the video-only demo

1. **Move the previous demo's generated recipe out of the way.** If
   `/media/mixer.demo.json` is left over from `mixer_audio` and all assets are
   already cached (`Assets ready in 0.0 s`), a failed start's recovery path
   re-launches that stored recipe — so the video-only demo ends up running 72
   `has_audio` sources with no `audio_router_node`, and its preheat wedges on
   `mixer_trans_out`. On the first swap this hides, because generating
   `synthetic_v5_*` also rewrites the recipe. Check what actually started:

   ```sh
   python3 -I -c 'import json;s=json.load(open("media/mixer.demo.json"))["sources"];
   print(sum(1 for x in s if x.get("has_audio")), sum(1 for x in s if x.get("decoded_format")))'
   # video-only expects: 0 72
   ```

2. **Keep `janus-preview` down until the mixer is on air.** This demo points
   `MIXER_STATE_URL` at `/api/outputs`, which goes through
   `bridge._query_static()` → `mixer.scenes` + `mixer.settings` on the control
   port. During preheat that port blocks until `setReady()`, so the first call
   hits the bridge timeout and the reply stream desynchronizes by one: the next
   `mixer.settings` is answered with the *scenes* list and `/api/outputs` fails
   with `AttributeError: 'list' object has no attribute 'get'`. Setup reads that
   as `Mixer startup timed out`, kills a mixer that is already on air, and falls
   into the recovery of trap 1. `mixer_audio` polls `/api/state` instead, which
   never touches that path, which is why only this demo shows it.

   ```sh
   docker compose -f docker-compose/mixer/compose.yaml stop janus-preview
   docker compose -f docker-compose/mixer/compose.yaml up -d --force-recreate mixer
   # wait for "setup: Mixer ready: setup applied"
   docker compose -f docker-compose/mixer/compose.yaml up -d janus-preview
   ```

### Restarting the audio demo needs the preheat fix in the container

Going back to `mixer_audio` failed three times in a row with

```
mixer preheat timed out during transition warm-up; 1/1 edges have no frame, first=mixer_trans_out
```

That is exactly the failure fixed by `98792a9b` ("Keep per-source audio drained
while preheating"), which is **not** in the deployed image (built 14:47, before
the commit). The live container had been patched with `docker cp`, and
`docker compose up -d --force-recreate mixer` throws that patch away. After
re-copying the single file the demo starts again:

```sh
scp -i ~/.ssh/awsdev.pem pyplumber/mixer/application.py fedora@<HOST>:~/loadcmp/
ssh -i ~/.ssh/awsdev.pem fedora@<HOST> \
  'docker cp ~/loadcmp/application.py avplumber-mixer-mixer-1:/build/pyplumber/mixer/application.py \
   && docker restart avplumber-mixer-mixer-1'
```

`docker restart` keeps the copied file; `up -d --force-recreate` does not. A
rebuild (`demos/mixer_audio/scripts/build.sh`) is what makes it permanent.
