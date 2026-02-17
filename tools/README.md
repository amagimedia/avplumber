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
