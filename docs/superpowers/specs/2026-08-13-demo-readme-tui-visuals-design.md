# Playlist and replay README TUI visuals

## Goal

Make the playlist and replay demo READMEs easy to scan without losing their
technical reference value. Each README must show the real terminal interface
first, then explain its user-visible features in plain language.

## Scope

Update only the documentation and documentation assets for:

- `demos/playlist/`
- `demos/replay/`

Do not change either application's behavior, controls, graph, or tests.

## TUI captures

The visuals must be rendered by the actual `PlaylistTui` and `ReplayTui`
classes. They must not be manually reconstructed in HTML, SVG, or an image
editor. This ensures that labels, ordering, spacing, borders, colors, and
visible controls match the implementation.

Use the representative ready/playing states already constructed by the TUI
tests. Render playlist at 180 columns by 50 rows and replay at 180 columns by
42 rows, matching their existing test sizes. Capture the playlist main screen
and its real Add/Edit modal because the modal contains user-visible controls
that do not appear on the main screen. Do not add controls or status fields
that the applications do not render.

Store the generated SVG assets beside the relevant demo documentation so that
GitHub can display them without custom CSS or JavaScript. Give every image
useful alternative text.

## README structure

Both READMEs use this reader-first order:

1. Demo title.
2. Actual TUI visualization.
3. Plain-language feature description.
4. Requirements and shortest working setup/run sequence.
5. Controls and configuration reference.
6. Regression/testing instructions.
7. Implementation details and limitations.

The opening sections should define unfamiliar terms when they first matter and
should tell the reader what the demo does, what they need, how to start it, and
what they should see. Detailed graph topology, fixture mechanics, seek-sidecar
formats, and acceptance-gate behavior remain available later in the document
for technical readers.

## Accuracy rules

Document every user-visible control exposed by the current TUI code. Include
CLI options when they are necessary to run or configure the demo, but avoid
turning the opening feature description into an option dump. Preserve explicit
limitations such as NVIDIA/CUDA requirements, video-only output, the existing
Janus mountpoint requirement, and omitted production features.

The obsolete playlist HTML mockup must not be cited as the interface source of
truth. The Python TUI implementations and their tests are authoritative.

## Verification

- Generate each SVG by running the real Textual app in its headless test mode.
- Confirm that the embedded asset paths resolve from their README.
- Compare visible labels and controls against the corresponding `player.py`.
- Run the playlist and replay TUI tests after the documentation work.
- Review both READMEs from top to bottom for a visualization-first order,
  concise feature description, complete run instructions, and preserved
  limitations.
