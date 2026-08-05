# FFmpeg Patch Stack

This directory contains the public patch stack for the FFmpeg variant used by
avplumber neural-net and CUDA workflows.

## Base

- Upstream repository: `https://github.com/FFmpeg/FFmpeg`
- Upstream tag: `n7.1.5`
- Upstream commit used for validation: `3a0867c2bfda4a4d4309ca1a8cbdc6175e67f587`

## Source

The legacy patches were originally exported from the private remote branch:

- branch: `n7.1.3-tellyodev`

The patch series is intended to be applied in numeric order onto a clean public
FFmpeg checkout at tag `n7.1.5`.

The current exported stack is compacted to keep CI churn manageable:

- patches `0001` through `0015` preserve the original functional commit split
- patch `0016-github-actions.patch` squashes the original GitHub Actions-only
  commits from the private branch into one CI patch
- patches `0017` onward preserve the later functional FFmpeg changes

## Apply

Example:

```bash
git clone --branch n7.1.5 --depth 1 https://github.com/FFmpeg/FFmpeg clean-ffmpeg
cd clean-ffmpeg
git am /path/to/avplumber/deps/ffmpeg-patches/*.patch
```

## Validation Target

Host validation path:

- `<ffmpeg-checkout>`

Install prefix used by the current validation plan:

- `/usr/local`

## Configure Contract

Validation is expected to use the exact configure line from the host custom
FFmpeg install being compared.

The goal is parity with the current custom FFmpeg install at:

- `/usr/local`
