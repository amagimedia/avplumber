"""Run the transition's metadata selection without requiring a CUDA device."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_transition_metadata(tmp_path):
    if not shutil.which("cc") or not shutil.which("pkg-config"):
        pytest.skip("C compiler and libavutil development files required")
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", "libavutil"], capture_output=True, text=True)
    if flags.returncode:
        pytest.skip("libavutil development files required")
    patch = (Path(__file__).resolve().parents[1] /
             "deps/ffmpeg/9/0021-avfilter-transition-source-metadata.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    start = added.index("static int transition_metadata")
    helper = added[start:added.index("\n}", start) + 2]
    program = r'''
#include <assert.h>
#include <string.h>
#include <libavutil/avstring.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#define TRANSITION_MODE_DIP 5
 typedef struct { int transition_mode; float alpha_coef; char *metadata_prefix; } TransitionCUDAContext;
''' + helper + r'''
int main(void) {
    TransitionCUDAContext ctx = {0, .5f, "avp.source."};
    AVFrame *a = av_frame_alloc(), *b = av_frame_alloc();
    av_dict_set(&a->metadata, "avp.source.a", "1:10:20", 0);
    av_dict_set(&a->metadata, "avp.source.shared", "1:11:20", 0);
    av_dict_set(&a->metadata, "color", "unchanged", 0);
    av_dict_set(&b->metadata, "avp.source.b", "2:22:20", 0);
    av_dict_set(&b->metadata, "avp.source.shared", "1:12:20", 0);
    for (int mode = 0; mode <= 5; mode += 5) {
        ctx.transition_mode = mode;
        for (int step = 0; step <= 4; ++step) {
            ctx.alpha_coef = step / 4.f;
            AVFrame *out = av_frame_alloc();
            assert(av_frame_copy_props(out, a) == 0);
            assert(transition_metadata(&ctx, out, b) == 0);
            int main = mode == 5 ? step < 2 : step < 4;
            int overlay = mode == 5 ? step > 2 : step > 0;
            assert(!!av_dict_get(out->metadata, "avp.source.a", NULL, 0) == main);
            assert(!!av_dict_get(out->metadata, "avp.source.b", NULL, 0) == overlay);
            const AVDictionaryEntry *shared = av_dict_get(out->metadata, "avp.source.shared", NULL, 0);
            if (main && overlay) assert(!strcmp(shared->value, "1:11:20;1:12:20"));
            if (!main && !overlay) assert(!shared);
            assert(!strcmp(av_dict_get(out->metadata, "color", NULL, 0)->value, "unchanged"));
            av_frame_free(&out);
        }
    }
    av_frame_free(&a); av_frame_free(&b);
}
'''
    source = tmp_path / "transition.c"
    source.write_text(program)
    binary = tmp_path / "transition"
    subprocess.run(["cc", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary), *flags.stdout.split()], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
