"""GPU 0 as nvidia-smi reports it, with its NVDEC and NVENC counts from NVIDIA's support matrix
(gpu_engines.json): what Setup sizes a show by on a GPU it has no measurements for. Standard
library only; `python3 demos/mixer/gpu_spec.py` prints the instance type for host scripts."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess

QUERY = "name,compute_cap,memory.total,memory.free"
ENGINES_FILE = Path(__file__).with_name("gpu_engines.json")


def name_key(name):
    """nvidia-smi's "NVIDIA GeForce RTX 4060 Ti" and the matrix's "GeForce RTX 4060TI" alike: no vendor,
    die or memory-size suffix, no "GPU", spaces or punctuation."""
    name = re.sub(r"\(.*?\)|\b\d+\s*gb\b|\bnvidia\b|\btesla\b|\bgpu\b", "", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def _engines():
    table = {}
    for board, row in json.loads(ENGINES_FILE.read_text())["gpus"].items():
        for alias in board.split("/"):   # "NVIDIA T400 / T400 4GB"
            table.setdefault(name_key(alias), row)
    return table


ENGINES = _engines()


@dataclass(frozen=True)
class GpuSpec:
    name: str
    compute_cap: str
    total_mib: int
    free_mib: int
    nvdec: int
    nvenc: int
    known: bool   # found in gpu_engines.json; otherwise one NVDEC and one NVENC are assumed

    @property
    def ada_or_newer(self):
        try:
            return float(self.compute_cap) >= 8.9
        except ValueError:
            return False


def parse(text):
    """One nvidia-smi CSV line of QUERY, or None for none, several GPUs or unavailable fields."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4 or not fields[2].isdigit() or not fields[3].isdigit():
        return None
    name, compute_cap, total, free = fields
    row = ENGINES.get(name_key(name))
    return GpuSpec(name, compute_cap, int(total), int(free),
                   row["nvdec"] if row else 1, row["nvenc"] if row else 1, row is not None)


def probe(run=None):
    """MIXER_GPU when set (the host's line, passed to containers without a GPU such as the seed),
    otherwise GPU 0, the one desktop.sh gives Chromium and the mixer. None when unavailable."""
    if os.environ.get("MIXER_GPU"):
        return parse(os.environ["MIXER_GPU"])
    try:
        out = (run or subprocess.check_output)(
            ["nvidia-smi", "-i", "0", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse(out)


def instance_type(spec):
    """A T4 or an L4 keeps its measured profile; any other GPU, or none found, the scaled GeForce one."""
    return {"t4": "tesla_t4", "l4": "nvidia_l4_cuarray"}.get(name_key(spec.name) if spec else "", "geforce")


if __name__ == "__main__":
    print(instance_type(probe()))
