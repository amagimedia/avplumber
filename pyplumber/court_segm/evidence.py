"""Evidence extraction from segmentation masks and detection metadata.

Everything that turns raw YOLO outputs (seg mask planes plus player/hoop
boxes) into fit-ready evidence: cleaned class masks, the synthetic 3-pt mask
from the floor hole, boundary/edge point extraction, and the close-up guard.

This module owns nearly all of the per-pixel CPU work (272x272 mask grids
every frame) — the top CUDA/CuPy port candidates live here; see
doc/court_calibration_realtime_port.md.
"""

import json
import struct

import numpy as np
from skimage.measure import find_contours

from pyplumber.court_segm import cuda as _gpu

AV_FRAME_DATA_YOLO_SEG_MASKS = 0x59534D00
AV_FRAME_DATA_COURT_LUMA = 0x4C554D41


class EvidenceMixin:
    """Mask/metadata parsing and evidence extraction for CourtCalibrationNode."""

    # --- input parsing -----------------------------------------------------

    def _read_masks(self, frame):
        want = AV_FRAME_DATA_YOLO_SEG_MASKS + self.seg_slot * 2
        for sd in frame.side_data:
            if int(sd.type) != want:
                continue
            data = bytes(sd.data)
            if len(data) < 16:
                return None
            num, w, h, _ = struct.unpack_from("<IIII", data, 0)
            need = 16 + num * w * h * 4
            if num == 0 or len(data) < need:
                return None
            arr = np.frombuffer(data, dtype=np.float32, count=num * w * h, offset=16)
            return arr.reshape(num, h, w)
        return None

    def _read_luma(self, frame):
        """Packed uint8 luma plane of the model-input frame, attached by
        court_seg_evidence_cuda (emit_luma) — painted-line snap evidence."""
        for sd in frame.side_data:
            if int(sd.type) != AV_FRAME_DATA_COURT_LUMA:
                continue
            data = bytes(sd.data)
            if len(data) < 16:
                return None
            w, h, _, _ = struct.unpack_from("<IIII", data, 0)
            if w == 0 or h == 0 or len(data) < 16 + w * h:
                return None
            return np.frombuffer(data, dtype=np.uint8, count=w * h,
                                 offset=16).reshape(h, w)
        return None

    def _read_json(self, frame, key):
        try:
            return json.loads(frame.metadata[key])
        except Exception:
            return None

    def _max_player_height_frac(self, frame):
        """Tallest Player/Ref bbox as a fraction of the model frame height."""
        md = self._read_json(frame, self.player_metadata_key)
        if not md:
            return 0.0
        mh = md.get("model_height", 544.0)
        best = 0.0
        for det in md.get("detections", []):
            if det.get("label") in ("Player", "Ref") and "xyxy" in det:
                best = max(best, (det["xyxy"][3] - det["xyxy"][1]) / mh)
        return best

    def _hoop_source_px(self, frame):
        md = self._read_json(frame, self.player_metadata_key)
        if not md:
            return None
        best, best_conf = None, -1.0
        for det in md.get("detections", []):
            if det.get("label") != "Hoop":
                continue
            conf = det.get("conf", 0.0)
            if conf > best_conf and "xyxy" in det:
                best, best_conf = det["xyxy"], conf
        if best is None:
            return None
        mw = md.get("model_width", 960.0)
        mh = md.get("model_height", 544.0)
        return (
            (best[0] + best[2]) * 0.5 * frame.width / mw,
            (best[1] + best[3]) * 0.5 * frame.height / mh,
        )

    # --- mask cleanup / synthesis (per-pixel heavy: CUDA candidates) --------

    def _clean_region_mask(self, m):
        """Enforce the 3-pt region's known topology: a single filled blob.

        The seg coefficients leak the court class into the arc interior and
        scatter low specks across the floor. Inside the arc only the 3-pt
        region is physically possible, so: threshold, keep the largest
        connected component (drops far-field specks), then fill enclosed holes
        (restores court-into-3pt bleed). Returns a cleaned float mask whose
        0.5 contour is the painted arc/straights, not a ragged blob edge.
        Non-destructive: float values inside the kept+filled region are
        preserved for sub-pixel contour interpolation; everything else is 0.
        """
        return _gpu.clean_region_mask(m, self.mask_threshold)

    def _class_planes(self, frame, masks):
        """Returns (court mask, list of 3pt-region masks)."""
        md = self._read_json(frame, self.seg_metadata_key)
        labels = []
        if md:
            labels = [d.get("label", "") for d in md.get("detections", [])]
        dets = md.get("detections", []) if md else []
        model_w = md.get("model_width", 960.0) if md else 960.0
        model_h = md.get("model_height", 544.0) if md else 544.0
        court, tpls = None, []
        self._tpl_conf = 0.0
        for i in range(masks.shape[0]):
            label = labels[i] if i < len(labels) else ""
            m = masks[i]
            if label == "three point line":
                if i < len(dets):
                    self._tpl_conf = max(self._tpl_conf,
                                         float(dets[i].get("conf", 0.0)))
                # The C++ mask assembly does not crop instance masks to their
                # detection box (ultralytics does), so coefficients activate
                # across the whole floor. Crop here to restore the instance.
                if i < len(dets) and "xyxy" in dets[i]:
                    x1, y1, x2, y2 = dets[i]["xyxy"]
                    mh, mw = m.shape
                    mx1 = max(0, int(x1 / model_w * mw) - 2)
                    my1 = max(0, int(y1 / model_h * mh) - 2)
                    mx2 = min(mw, int(x2 / model_w * mw) + 3)
                    my2 = min(mh, int(y2 / model_h * mh) + 3)
                    cropped = np.zeros_like(m)
                    cropped[my1:my2, mx1:mx2] = m[my1:my2, mx1:mx2]
                    m = cropped
                if self.clean_tpl_mask:
                    m = self._clean_region_mask(m)
                tpls.append(m)
            elif label == "basketball-court" or not labels:
                court = m if court is None else np.maximum(court, m)
        return court, tpls

    def _synth_tpl_from_court(self, court):
        """When the 3-pt class is absent, the court class usually still
        excludes the 3-pt interior — the enclosed un-fired region inside the
        court mask IS the 3-pt region, and its boundary is the painted arc.
        Frame borders are closed first so a region clipped by the frame edge
        still counts as enclosed; candidate components must be mostly
        surrounded by court pixels (crowd/apron gaps are bounded by the
        border or one-sided). Returns a synthetic float mask or None."""
        if court is None:
            return None
        out = _gpu.synth_holes(court, self.mask_threshold,
                               self.min_region_area)
        return self._clean_region_mask(out) if out is not None else None

    def _pick_tpl(self, tpls, hoop, w, h):
        """Both 3-pt regions can be visible in a wide shot; the model is one
        region — pick the plane nearest the hoop, else the largest."""
        best, best_key = None, None
        for m in tpls:
            binm = m >= self.mask_threshold
            area = float(binm.mean())
            if area < 0.01:
                continue
            if hoop is not None:
                mh, mw = m.shape
                ys, xs = np.nonzero(binm)
                cx = xs.mean() / mw * w
                cy = ys.mean() / mh * h
                key = np.hypot(cx - hoop[0], cy - hoop[1])
                better = best_key is None or key < best_key
            else:
                key = -area
                better = best_key is None or key < best_key
            if better:
                best, best_key = m, key
        return best

    # --- mask <-> source coordinate mapping ----------------------------------

    def _mask_idx_to_norm(self, cols, rows, mw, mh):
        """Mask index coords (find_contours/nonzero convention: integer index
        = pixel center) -> normalized source coords, undoing the model-frame
        letterbox the mask covers."""
        xn = ((np.asarray(cols, dtype=float) + 0.5) * (self.mask_model_w / mw)
              - self.mask_pad_x) / self.mask_content_w
        yn = ((np.asarray(rows, dtype=float) + 0.5) * (self.mask_model_h / mh)
              - self.mask_pad_y) / self.mask_content_h
        return xn, yn

    def _norm_to_mask_idx(self, xn, yn, mw, mh):
        """Inverse of _mask_idx_to_norm (continuous index coords)."""
        ci = (np.asarray(xn, dtype=float) * self.mask_content_w
              + self.mask_pad_x) * (mw / self.mask_model_w) - 0.5
        ri = (np.asarray(yn, dtype=float) * self.mask_content_h
              + self.mask_pad_y) * (mh / self.mask_model_h) - 0.5
        return ci, ri

    # --- boundary / edge point extraction ------------------------------------

    def _mask_boundary_points(self, mask, w, h, min_longest=40, min_len=20):
        contours = find_contours(mask, self.mask_threshold)
        if not contours:
            return None
        contours.sort(key=len, reverse=True)
        if len(contours[0]) < min_longest:
            return None
        mh, mw = mask.shape
        all_pts = []
        for c in contours:
            if len(c) < min_len:
                continue
            keep = (
                (c[:, 1] > self.border_margin_px)
                & (c[:, 1] < mw - 1 - self.border_margin_px)
                & (c[:, 0] > self.border_margin_px)
                & (c[:, 0] < mh - 1 - self.border_margin_px)
            )
            pts = c[keep]
            if len(pts):
                xn, yn = self._mask_idx_to_norm(pts[:, 1], pts[:, 0], mw, mh)
                all_pts.append(np.stack([xn * w, yn * h], axis=1))
        if not all_pts:
            return None
        return np.vstack(all_pts)

    def _tpl_boundary_points(self, tpl, w, h):
        """Sub-pixel boundary of the 3-pt region in source px, border-clipped."""
        return self._mask_boundary_points(tpl, w, h)

    def _court_boundary_points(self, court, w, h):
        """Court/non-court contour in source px.

        This includes the visible 3pt hole boundary even when that region is
        open/clipped and cannot be synthesized as a filled template mask.
        """
        return self._mask_boundary_points(court, w, h, min_longest=30,
                                          min_len=12)

    def _baseline_edge_points(self, court, w, h, hoop_on_left):
        """Per-row extreme x of the court mask on the hoop side = the visible
        baseline edge (the court outline is near-perfect; use it directly).
        Frame-clipped rows are excluded. Normalized coords."""
        mh, mw = court.shape
        binm = court >= self.mask_threshold
        pts = []
        rows = np.nonzero(binm.any(axis=1))[0]
        if not len(rows):
            return None
        # skip the lowest 25% of court rows: that edge is the crowd cut
        max_row = rows.min() + int((rows.max() - rows.min()) * 0.75)
        for r in rows:
            if r > max_row or r < 2:
                continue
            cols = np.nonzero(binm[r])[0]
            x = cols.min() if hoop_on_left else cols.max()
            if x < 3 or x > mw - 4:
                continue  # frame-clipped
            xn, yn = self._mask_idx_to_norm(x, r, mw, mh)
            pts.append((float(xn), float(yn)))
        return np.array(pts) if len(pts) >= 10 else None

    def _court_bottom_edge(self, court, w, h):
        """Per-column lowest court-mask row = the court/crowd cut (source
        px). NOT the painted near sideline — but the sideline must sit just
        above it, which bounds the otherwise-unconstrained near side."""
        mh, mw = court.shape
        binm = court >= self.mask_threshold
        cols = np.nonzero(binm.any(axis=0))[0]
        pts = []
        for c in cols:
            if c < 3 or c > mw - 4:
                continue
            r = mh - 1 - int(np.argmax(binm[::-1, c]))
            if r >= mh - 2:
                continue  # clipped at frame bottom
            xn, yn = self._mask_idx_to_norm(c, r, mw, mh)
            pts.append((float(xn) * w, float(yn) * h))
        return np.array(pts) if len(pts) >= 8 else None

    def _court_top_edge(self, court, w, h):
        mh, mw = court.shape
        binm = court >= self.mask_threshold
        cols = np.nonzero(binm.any(axis=0))[0]
        pts = []
        for c in cols:
            if c < 3 or c > mw - 4:
                continue
            r = int(np.argmax(binm[:, c]))
            if r <= 1:
                continue  # clipped at frame top
            xn, yn = self._mask_idx_to_norm(c, r, mw, mh)
            pts.append((float(xn) * w, float(yn) * h))
        return np.array(pts) if len(pts) >= 8 else None
