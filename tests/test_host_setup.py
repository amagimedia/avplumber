"""docker-compose/mixer/host-setup.sh check with nvidia-smi, ldconfig and docker stubbed: the
driver gate and the Docker items. The /sys and /usr items read this machine and are not asserted."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docker-compose/mixer/host-setup.sh"


def check(tmp_path, driver="615.71.09", runtime="runc nvidia ", kernels=(), built=()):
    stubs, boot = tmp_path / "bin", tmp_path / "boot"
    stubs.mkdir()
    boot.mkdir()
    for kernel in kernels:
        (boot / f"vmlinuz-{kernel}").write_text("")
    scripts = {
        "modinfo": f'case " {" ".join(built)} " in *" $2 "*) exit 0 ;; *) exit 1 ;; esac',
        "nvidia-smi": f'echo "NVIDIA GeForce RTX 5080, {driver}"',
        "ldconfig": 'echo "  libnvcuvid.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1"',
        "docker": f'case "$*" in "compose version") exit 0 ;; info) exit 0 ;; *Runtimes*) echo "{runtime}" ;; esac',
    }
    for name, body in scripts.items():
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n")
        (stubs / name).chmod(0o755)
    return subprocess.run(["bash", str(SCRIPT), "check"], capture_output=True, text=True, timeout=30,
                          env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "HOST_SETUP_BOOT": str(boot)})


def test_a_current_driver_and_toolkit_pass_their_items(tmp_path):
    out = check(tmp_path).stdout
    assert "ok    NVIDIA driver 615.71.09 on NVIDIA GeForce RTX 5080" in out
    assert "ok    NVDEC library" in out and "FIX   libnvidia-encode.so.1 missing" in out
    assert "ok    NVIDIA Container Toolkit runtime" in out and "ok    Docker Compose v2" in out


@pytest.mark.parametrize("driver", ["595.58.03", "610.10"])
def test_an_older_driver_is_refused_with_the_fix(tmp_path, driver):
    result = check(tmp_path, driver=driver)
    assert result.returncode == 1
    assert f"NVIDIA driver R615 or newer is needed (found {driver})" in result.stdout


def test_docker_without_the_nvidia_runtime_names_the_command(tmp_path):
    out = check(tmp_path, runtime="io.containerd.runc.v2 runc ").stdout
    assert "FIX   Docker has no nvidia runtime" in out and "nvidia-ctk runtime configure" in out


def test_usage(tmp_path):
    assert subprocess.run(["bash", str(SCRIPT), "frobnicate"], capture_output=True).returncode == 2


def test_a_newer_kernel_without_the_nvidia_module_is_flagged(tmp_path):
    out = check(tmp_path, kernels=("7.0.0-1011-gcp", "7.0.0-1014-gcp"), built=("7.0.0-1011-gcp",)).stdout
    assert "FIX   kernel 7.0.0-1014-gcp has no NVIDIA module, so a reboot loses the GPU" in out


def test_the_boot_kernel_with_its_module_passes(tmp_path):
    out = check(tmp_path, kernels=("7.0.0-1011-gcp",), built=("7.0.0-1011-gcp",)).stdout
    assert "ok    NVIDIA module for the boot kernel 7.0.0-1011-gcp" in out
