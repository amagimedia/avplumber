#!/usr/bin/env bash
#
# Verify that avplumber's embedded CUDA module image path really produces and
# loads fatbins for the configured GPU architectures.
#
# What this checks:
#   1. make emits a CUDA fatbin target for a representative avplumber .cu file.
#   2. cuobjdump sees the requested sm_* native slices in that fatbin.
#   3. On a GPU host, cuModuleLoadDataEx can load the same fatbin with PTX JIT
#      disabled. That proves the current GPU is using a native fatbin slice.
#
# Run from the avplumber repository root:
#   tools/verify_cuda_fatbin.sh
#
# On L4 this validates sm_89 at load time; on T4 it validates sm_75 at load
# time. The static cuobjdump check validates all requested slices.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TARGET="${1:-objs/src/nodes/neural_net/draw/draw_bbox.ptx}"
ARCHS="${CUDA_FATBIN_ARCHS:-75 80 86 89}"

make "${TARGET}" \
  HAVE_CUDA=1 HAVE_NVCC=1 NEURAL_NET=1 \
  CUDA_MODULE_IMAGE_FORMAT=fatbin \
  CUDA_FATBIN_ARCHS="${ARCHS}"

if [[ ! -s "${TARGET}" ]]; then
  echo "verify_cuda_fatbin: ERROR: missing fatbin target ${TARGET}" >&2
  exit 1
fi

if command -v cuobjdump >/dev/null 2>&1; then
  dump="$(cuobjdump -lelf "${TARGET}" 2>&1 || true)"
  for arch in ${ARCHS}; do
    if ! grep -q "sm_${arch}" <<<"${dump}"; then
      echo "verify_cuda_fatbin: ERROR: ${TARGET} does not contain sm_${arch}" >&2
      echo "${dump}" >&2
      exit 2
    fi
  done
  echo "verify_cuda_fatbin: cuobjdump found native slices: ${ARCHS}"
else
  echo "verify_cuda_fatbin: WARN: cuobjdump not found; static slice inspection skipped" >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "verify_cuda_fatbin: nvidia-smi not found; runtime load check skipped"
  exit 0
fi

gpu_csv="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits -i 0 | head -1 || true)"
if [[ -z "${gpu_csv}" ]]; then
  echo "verify_cuda_fatbin: nvidia-smi returned no GPU; runtime load check skipped"
  exit 0
fi

CUDA_DISABLE_PTX_JIT=1 python3 - "${TARGET}" <<'PY'
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def check(res: int, name: str, libcuda: ctypes.CDLL) -> None:
    if res == 0:
        return
    message = ctypes.c_char_p()
    get_error = getattr(libcuda, "cuGetErrorString", None)
    if get_error is not None and get_error(res, ctypes.byref(message)) == 0 and message.value:
        detail = message.value.decode("utf-8", "replace")
    else:
        detail = f"CUDA error {res}"
    raise SystemExit(f"verify_cuda_fatbin: ERROR: {name} failed: {detail}")


fatbin = Path(sys.argv[1]).read_bytes()
libcuda = ctypes.CDLL("libcuda.so.1")

CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p

check(libcuda.cuInit(0), "cuInit", libcuda)
device = CUdevice()
check(libcuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet", libcuda)
ctx = CUcontext()
check(libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device), "cuCtxCreate", libcuda)
try:
    module = CUmodule()
    image = ctypes.create_string_buffer(fatbin)
    check(
        libcuda.cuModuleLoadDataEx(
            ctypes.byref(module),
            ctypes.cast(image, ctypes.c_void_p),
            0,
            None,
            None,
        ),
        "cuModuleLoadDataEx(CUDA_DISABLE_PTX_JIT=1)",
        libcuda,
    )
    check(libcuda.cuModuleUnload(module), "cuModuleUnload", libcuda)
finally:
    check(libcuda.cuCtxDestroy_v2(ctx), "cuCtxDestroy", libcuda)

print("verify_cuda_fatbin: runtime load succeeded with CUDA_DISABLE_PTX_JIT=1")
PY

echo "verify_cuda_fatbin: OK current_gpu='${gpu_csv}' target='${TARGET}'"
