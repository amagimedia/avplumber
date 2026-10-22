import subprocess


def test_audio_mixer_core(cpp_binary):
    binary = cpp_binary("test_audio_mixer_core", libs=("libavutil",))
    subprocess.run([str(binary)], check=True, timeout=120)
