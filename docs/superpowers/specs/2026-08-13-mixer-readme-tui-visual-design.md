# Mixer README TUI visual and reader-first structure

## Goal

Give the generic mixer demo the same documentation structure as the playlist
and replay demos: show the real terminal interface first, then explain its
features in plain language before presenting setup and implementation details.

## Scope

Update the mixer README, add its documentation image under `demos/mixer/`, and
make the smallest Textual layout correction needed for the existing settings
controls to be visible at the tested terminal size. Do not change mixer graph
behavior or the control protocol.

## TUI capture

Generate the visual with the actual `MixerTui` class and Textual's SVG renderer.
Do not reconstruct the interface by hand in HTML, SVG, or an image editor.

Render at 160 columns by 45 rows, matching the existing TUI test. Use the
test's deterministic connected state:

- control connection at `127.0.0.1:7777`;
- `fullscreen_0` on Program;
- `grid_2_page_0` on Preview;
- `grid_4_page_0` available in the scene strip;
- transition mode idle;
- Direct mode off;
- fade duration 0.5 seconds; and
- wipe style `wipe_left`.

This state shows the useful production control surface rather than the brief
startup Disconnected state. Store the generated SVG beside the mixer
documentation and embed it immediately below the README title with useful
alternative text.

## TUI settings-row correction

The current `#transition_status` Static inherits full available width inside
the horizontal settings container. Its computed width grows with the terminal,
so the already-implemented transition-duration and wipe-style fields are always
placed beyond the screen edge. Set only this status widget to content-sized
width (`width: auto`). Preserve all labels, controls, actions, and behavior.

Extend the TUI test to assert that the transition status, duration input, wipe
style input, and the existing buttons all fit within the 160-by-45 screen. This
turns the renderer-discovered layout problem into a regression check.

## README structure

Use this order:

1. Demo title.
2. Actual TUI visualization.
3. Plain-language feature description.
4. Requirements.
5. Start the mixer backend.
6. Start and operate the separate TUI.
7. Output and Janus configuration.
8. Layout reference.
9. Docker workflow.
10. Tests and live acceptance checks.
11. Zero-copy and preheating implementation notes.

The feature description must cover only controls implemented by `MixerTui`:
Program and Preview buses; the scene strip; fullscreen and paged 2/4/8/16-box
layouts; Cut, Fade, and CUDA Wipe; Direct mode; Reconnect; transition duration;
wipe style; and the implemented keyboard bindings. Explain that normal scene
selection loads Preview, Direct mode cuts selections to Program, and F1-F9
always cut directly.

Keep current limitations explicit: video only, portrait 1080x1920 output, no
automatic selection or source-specific analysis, an existing Janus mountpoint,
NVENC output, and zero-copy CUDA with no CPU upload/download workaround.

## Accuracy and maintenance

Treat `demos/mixer/mixer.py`, `demos/mixer/tui.py`, and their tests as the
sources of truth. Preserve all useful CLI configuration, layout dimensions,
container instructions, smoke-test instructions, and remote NVIDIA acceptance
requirements. Move advanced graph construction and preheating details later in
the README so they do not obstruct the first-run path.

## Verification

- Generate the SVG through the running Textual app in headless test mode.
- Visually inspect the generated capture.
- Confirm every visible label and documented keyboard control against
  `tui.py`.
- Confirm the settings-row controls fit inside the tested 160-by-45 screen.
- Confirm documented mixer and TUI CLI defaults against their argument parsers.
- Confirm the README image path resolves.
- Run `python3 -m pytest -q demos/mixer/tests`.
- Do not attempt a local CUDA runtime check when `nvidia-smi` is unavailable;
  retain the documented remote acceptance requirement instead.
