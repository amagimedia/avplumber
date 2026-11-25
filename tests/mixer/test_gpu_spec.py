import json
import subprocess

import pytest

from demos.mixer import gpu_spec


def test_a_known_card_takes_its_engines():
    spec = gpu_spec.parse("NVIDIA GeForce RTX 5080, 12.0, 16303, 15800")
    assert spec == gpu_spec.GpuSpec("NVIDIA GeForce RTX 5080", "12.0", 16303, 15800, 2, 2, True)
    assert spec.ada_or_newer


@pytest.mark.parametrize("name, nvdec, nvenc", [
    ("NVIDIA GeForce RTX 4060 Ti", 1, 1), ("NVIDIA GeForce RTX 4090", 1, 2), ("NVIDIA L4", 4, 2),
    ("Tesla T4", 2, 1), ("NVIDIA GeForce RTX 5080 Laptop GPU", 2, 2), ("NVIDIA T400 4GB", 1, 1),
])
def test_names_match_the_matrix_spelling(name, nvdec, nvenc):
    spec = gpu_spec.parse(f"{name}, 8.9, 16000, 15000")
    assert (spec.nvdec, spec.nvenc, spec.known) == (nvdec, nvenc, True)


def test_an_unknown_card_assumes_one_engine_each():
    spec = gpu_spec.parse("NVIDIA GeForce RTX 9999, 13.0, 32000, 31000")
    assert (spec.nvdec, spec.nvenc, spec.known) == (1, 1, False)


@pytest.mark.parametrize("text", ["", "[N/A]", "NVIDIA L4, 8.9, [N/A], [N/A]",
                                  "NVIDIA L4, 8.9, 23034, 22000\nNVIDIA L4, 8.9, 23034, 22000"])
def test_unavailable_or_several_gpus_give_none(text):
    assert gpu_spec.parse(text) is None


def test_probe_prefers_the_hosts_line_and_survives_a_missing_nvidia_smi(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError("nvidia-smi")
    monkeypatch.delenv("MIXER_GPU", raising=False)
    assert gpu_spec.probe(missing) is None
    monkeypatch.setenv("MIXER_GPU", "NVIDIA GeForce RTX 3060, 8.6, 12288, 11000")
    assert gpu_spec.probe(missing).name == "NVIDIA GeForce RTX 3060"


@pytest.mark.parametrize("line, expected", [
    ("Tesla T4, 7.5, 15360, 15000", "tesla_t4"), ("NVIDIA L4, 8.9, 23034, 22000", "nvidia_l4_cuarray"),
    ("NVIDIA GeForce RTX 5080, 12.0, 16303, 15800", "geforce"), ("NVIDIA L40S, 8.9, 46068, 45000", "geforce"),
])
def test_t4_and_l4_keep_their_measured_profiles(line, expected):
    assert gpu_spec.instance_type(gpu_spec.parse(line)) == expected
    assert gpu_spec.instance_type(None) == "geforce"


def test_the_table_names_its_source_and_has_engine_counts():
    data = json.loads(gpu_spec.ENGINES_FILE.read_text())
    assert data["source"].startswith("https://developer.nvidia.com/") and data["read"]
    assert all(type(row["nvdec"]) is int and type(row["nvenc"]) is int for row in data["gpus"].values())
