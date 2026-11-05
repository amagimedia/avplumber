"""A missing/restarting audio_router node must not abort a take (AFV signaling is best-effort),
against the whole library."""
import subprocess


def test_missing_audio_router_does_not_abort_the_take(cpp_library_binary):
    subprocess.run([str(cpp_library_binary("test_mixer_audio_router_resilience"))], check=True, timeout=20)
