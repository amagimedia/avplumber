"""Court segmentation calibration internals (basketball tactical view).

Split of the former monolithic pyplumber/court_calibration.py along its
functional (and planned C++/CUDA port) boundaries:

- geometry: court model constants, homography helpers, physical PTZ camera
- evidence: masks/metadata -> fit-ready points (per-pixel heavy; CUDA candidates)
- matching: pre-rendered PTZ template grid + IoU camera search
- fitting:  segment assignment, sector consensus, PTZ least-squares refine
- temporal: ByteTrack-style PTZ filter, zones, publish/hold/rescue paths

The node class lives in pyplumber/court_calibration.py and composes the
mixins defined here.
"""
