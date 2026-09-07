"""Exercise the shared frame-selection contract without CUDA or FFmpeg."""
import pathlib
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def playout_binary(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mixer-playout")
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('no C++ compiler available')
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = tmp_path / 'mixer_playout'
    subprocess.run([
        compiler, '-std=c++17', '-O0', '-g', '-Wall', '-Wextra',
        '-I', str(root / 'src'), str(root / 'tests/cpp/test_mixer_playout.cpp'),
        '-o', str(binary),
    ], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("case", [
    "complete_jitter_plateau_is_not_rate_drift",
    "stale_burst_is_not_retimestamped_as_fresh",
    "rational_source_drift_does_not_amplify",
    "sparse_missing_paints_do_not_amplify_into_persistent_losses",
    "latency_cannot_exceed_retained_frames",
    "resumed_source_is_live_after_an_eof_marker",
    "presentation_phase_on_half_tick_keeps_every_frame",
    "route_reset_rejects_old_frames_still_in_upstream_edges",
    "prewarmed_slots_share_frame_ids_and_output_ticks",
    "scene_reload_discards_prewarm_frames_before_first_visible_frame",
    "stalled_input_does_not_block_healthy_inputs_or_replay_late_burst",
    "irregular_first_paints_do_not_shorten_the_playout_delay",
    "prewarm_waits_for_every_active_slot",
    "inactive_slot_reactivation_does_not_reuse_old_scene",
    "eof_drains_future_frames_before_finishing",
    "rational_clock_and_reference_lifetime",
    "invalid_parameters_fail_early",
    "preserves_presentation_timestamps_for_rate_conversion",
    "source_clock_gap_recovers_bounded_delay",
    "latency_and_backpressure",
    "burst_keeps_every_frame",
    "sixteen_independent_phases",
    "bounded_queue_counts_overflow",
    "missed_deadlines_do_not_catch_up_in_bursts",
])
def test_mixer_playout(playout_binary, case):
    result = subprocess.run([str(playout_binary), case], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
