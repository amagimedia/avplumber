"""The receipt counts successful encoded writes, never merely opened outputs."""
import json
import subprocess


def test_output_progress_is_atomic_throttled_and_counts_writes(cpp_binary, tmp_path):
    binary = cpp_binary('test_output_progress')
    result = subprocess.run([str(binary), str(tmp_path / 'progress.json')],
                            check=True, capture_output=True, text=True, timeout=10)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row['written_packets'] for row in rows] == [1, 1, 3, 4]
    assert [row['written_bytes'] for row in rows] == [100, 100, 600, 1000]
    assert [row['updated_at'] for row in rows] == [100, 100, 101.1, 101.4]
    assert all(row['first_packet_at'] == 100 for row in rows)


def test_fifo_submission_never_claims_network_delivery(cpp_binary, tmp_path):
    binary = cpp_binary('test_output_progress_submitted')
    result = subprocess.run([str(binary), str(tmp_path / 'submitted.json')],
                            check=True, capture_output=True, text=True, timeout=10)
    row = json.loads(result.stdout)
    assert row == {'delivery_semantics': 'submitted', 'submitted_packets': 2,
                   'submitted_bytes': 579, 'last_submit_at': 101, 'first_submit_at': 100}
    assert not any(key.startswith('written') for key in row)
