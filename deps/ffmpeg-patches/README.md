# FFmpeg Patch Stack

This directory contains the public patch stack for the FFmpeg variant used by
avplumber neural-net and CUDA workflows.

## Base

- Upstream repository: `https://github.com/FFmpeg/FFmpeg`
- Upstream tag: `n7.1.3`
- Upstream commit used for export: `f46e514491172d15bd74b4abb1814cd2f05a763e`

## Source

The patches were exported from the private remote branch:

- branch: `n7.1.3-tellyodev`

The patch series is intended to be applied in numeric order onto a clean public
FFmpeg checkout at tag `n7.1.3`.

## Apply

Example:

```bash
git clone --branch n7.1.3 --depth 1 https://github.com/FFmpeg/FFmpeg clean-ffmpeg
cd clean-ffmpeg
git am /path/to/avplumber/deps/ffmpeg-patches/*.patch
```

## Validation Target

Host validation path:

- `/home/fedora/clean-ffmpeg`

Install prefix used by the current validation plan:

- `/usr/local`

## Configure Contract

Validation is expected to use the exact host configure line captured in:

- `docs/superpowers/specs/2026-04-12-ffmpeg-public-patch-stack-design.md`

The goal is parity with the current custom FFmpeg install at:

- `/apps/ffmpeg`
