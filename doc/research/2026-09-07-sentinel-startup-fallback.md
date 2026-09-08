# Sentinel cold-start fallback behavior

Reviewed 2026-09-07 from repository implementation, documentation, examples,
and local git history. No implementation was changed.

## Conclusion

The current sentinel cannot be configured to **emit nothing until the first
valid input frame, then arm normal failover**. It has no first-input/armed
latch. An empty input at cold start eventually enters the same backup path as a
stream which started and then stalled.

For video, sentinel does not intrinsically generate black. It repeats a seeded
or last good frame for `freeze`, then emits the required configured backup
image/buffer. Consequently:

- without real input or `initial_picture_buffer`, cold-start fallback goes
  directly to the configured slate;
- an initial black frame occurs only if `initial_picture_buffer` or the regular
  backup contains black;
- `freeze: 0` bypasses repeated-frame output and goes directly to the slate; it
  does not suppress output.

The documented purpose is continuous output during missing input, including
silence for audio, last-frame freeze for video, and then a custom video slate.
[Sentinel documentation](/home/jp/git/avplumber/doc/NODES.md:298)

## State and timing

Startup state includes `last_success_ = true`, timeout/freeze durations, and
the last frame, but no state recording that a valid input has ever arrived.
[Startup members](/home/jp/git/avplumber/src/nodes/sentinel.cpp:500)

While `last_success_` is true, the input `peek()` waits for `timeout`; after an
unsuccessful poll, later polls use 22 ms, or 150 ms while the output queue is
full. [Polling logic](/home/jp/git/avplumber/src/nodes/sentinel.cpp:679) An empty
poll enters backup handling, initializes `next_ts_` from `start_ts` when needed,
and marks the poll unsuccessful. On later polls, `!last_success_` enables backup
generation even though no valid input frame has ever been seen.
[Cold-start backup path](/home/jp/git/avplumber/src/nodes/sentinel.cpp:895)

With a fresh, otherwise unused correction group, the first timeout bootstraps
its clock at `start_ts`. The clock reports that same timestamp for wall-clock
gaps below two seconds, then advances, so actual first cold-start output may be
roughly `timeout + 2 seconds`, rather than exactly `timeout`. A correction group
already clocked by another sentinel can behave differently. This timing detail
does not change the conclusion: fallback eventually starts without prior input.
[Shared-clock bootstrap](/home/jp/git/avplumber/src/nodes/sentinel.cpp:114)

After a genuine frame is output, sentinel clears card state and saves its PTS
and frame. A later stall can therefore repeat that real frame until `freeze`
expires. [Valid-input handling](/home/jp/git/avplumber/src/nodes/sentinel.cpp:860)
The backup selector requires a complete last frame and valid last non-card PTS;
otherwise it immediately chooses the configured backup/slate.
[Backup selection](/home/jp/git/avplumber/src/nodes/sentinel.cpp:565)

## Relevant options

- `timeout` only controls the initial blocking poll / ordinary stall delay; it
  is not an arming condition. The documented default is one second.
  [Option documentation](/home/jp/git/avplumber/doc/NODES.md:311)
- `freeze` controls how long an available last frame is repeated before the
  slate. It cannot defer a cold-start slate when no last frame exists.
  [Video options](/home/jp/git/avplumber/doc/NODES.md:338)
- `initial_picture_buffer` deliberately seeds the last-frame state at
  `start_ts`; documentation describes using it to show black instead of the
  slate initially. It substitutes one synthetic picture for another rather
  than suppressing output.
  [Initialization code](/home/jp/git/avplumber/src/nodes/sentinel.cpp:1017)
- `forward_start_shift` controls first-packet timestamp alignment. With it
  disabled, the constructor seeds `next_ts_`; with it enabled, the timeout path
  seeds `next_ts_` anyway. It therefore does not inhibit cold-start fallback.
  [Constructor behavior](/home/jp/git/avplumber/src/nodes/sentinel.cpp:946)
- `eof_passthrough: true` forwards an actual EOF and finishes. It has no effect
  on an empty live queue and cannot later arm failover.
  [EOF handling](/home/jp/git/avplumber/src/nodes/sentinel.cpp:879)
- `hold_until_ms` / `hold_until_iso` are undocumented creation parameters for an
  absolute wall-clock gate. Before the deadline, sentinel generates no backups
  **and silently drops valid input frames**; after the deadline, it permanently
  disables the gate regardless of whether input ever arrived.
  [Gate behavior](/home/jp/git/avplumber/src/nodes/sentinel.cpp:630)
  [Parameter parsing](/home/jp/git/avplumber/src/nodes/sentinel.cpp:1122)
  This can gate a known scheduled start, but it is not first-input activation.
- Video construction requires a backup. The implementation accepts
  `backup_image` or `backup_picture_buffer` and throws if neither yields a valid
  frame. [Video backup construction](/home/jp/git/avplumber/src/nodes/sentinel.cpp:407)
  There is a documentation mismatch: it names `backup_frame`, while source and
  examples use `backup_image`.
  [Documented name](/home/jp/git/avplumber/doc/NODES.md:342)
  [Example](/home/jp/git/avplumber/examples/video_recorder.avplumber:17)

## Practical implication

Meeting the exact requirement requires either external lifecycle orchestration
that starts sentinel/output only after input readiness, or a new explicit
sentinel state/option such as `wait_for_first_input`. Increasing `timeout` merely
delays both cold-start fallback and post-start failover; it does not encode the
required state transition.
