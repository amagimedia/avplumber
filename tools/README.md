# Tools

## waveform_console_demo.py

ANSI-art console waveform display for envelope files produced by the `write_audio_envelope` node.

**Usage**

```bash
python3 waveform_console_demo.py <path_to_envelope_dir> [options]
```

**Example**

```bash
# Record with write_audio_envelope (e.g. path /tmp/envelope), then:
python3 waveform_console_demo.py /tmp/envelope
python3 waveform_console_demo.py /tmp/envelope --level "1/25" --metric rms --width 60 --height 10
```

**Options**

- `--level KEY` – Level key (`duration_sec` string, e.g. `1/25`). Default: first level in index.
- `--metric {positive_peak,negative_peak,rms}` – Metric to draw (default rms)
- `--channel N` – Channel index; -1 = max across all channels (default)
- `--width`, `--height` – Display size in characters (default 72×12)
- `--no-ansi` – Disable ANSI color (plain ASCII)

The directory must contain `index.json` and the level binary files (e.g. `level_0.bin`) as written by `write_audio_envelope`.

## nsight/nsight.sh

NVIDIA Nsight Systems for an avplumber image: which CUDA kernels the GPU spends its time on. Nothing
is installed on the host; the profiler goes into a second image built over the one that runs.

**Usage**

```bash
tools/nsight/nsight.sh image <base-image> [<tag>]          # <base-image> plus the Nsight Systems CLI (default tag: <base-image>-nsys)
tools/nsight/nsight.sh summary <trace.nsys-rep> [<image>]  # GPU time per kernel and grid, launches per second, share of the window
```

`image` reads the base image's distribution and installs the newest `nsight-systems` package of
NVIDIA's CUDA repository for it (Fedora and RHEL-family images). A trace is the deployment's own
command with `nsys profile` in front of it, in that image:

```bash
docker compose run --rm --no-deps --entrypoint nsys <service> \
  profile -t cuda -s none --cpuctxsw=none --delay=60 --duration=40 \
  -o <mounted dir>/trace --force-overwrite true <the service's command and arguments>
```

Stop the regular service first; `--duration` ends the trace and stops the process with SIGTERM.
Compare the GPU load during the trace with the usual one before trusting the shares. TensorRT
inference runs as CUDA graphs: `summary` counts their launches, and their GPU time is what the
GPU's utilisation leaves over the kernels' share; `--cuda-graph-trace=node` lists the kernels inside
the graphs too, at a higher tracing cost. `DOCKER="sudo docker"` where docker needs it.
