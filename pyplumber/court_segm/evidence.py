"""Evidence extraction from segmentation masks and detection metadata.

Everything that turns raw YOLO outputs (seg mask planes, player/hoop boxes,
court-pose keypoints) into fit-ready evidence: cleaned class masks, the
synthetic 3-pt mask from the floor hole, spillover reconciliation, boundary
and edge point extraction, and the player-scale physics check.

This module owns nearly all of the per-pixel CPU work (272x272 mask grids
every frame) — the top CUDA/CuPy port candidates live here; see
doc/court_calibration_realtime_port.md.
"""

import json
import struct

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from pyplumber.court_segm.geometry import (
    COURT_W, COURT_H, _POSE_CANON, _apply_h,
)

AV_FRAME_DATA_YOLO_SEG_MASKS = 0x59534D00


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

    def _read_json(self, frame, key):
        try:
            return json.loads(frame.metadata[key])
        except Exception:
            return None

    def _parse_mid_pose(self, frame):
        """Confident MID-COURT pose keypoints only (quarter marks + center
        line x sidelines): (img_norm Nx2, court_norm Nx2) or (None, None).
        Used exclusively in mid-pan mode where the alternative is an
        unobservable pan — never at the court ends, which the segmentation
        machinery serves with far better precision."""
        md = self._read_json(frame, self.pose_metadata_key)
        if not md or not md.get("poses"):
            return None, None
        pose = max(md["poses"], key=lambda q: q.get("conf", 0.0))
        kpts = pose.get("keypoints", [])
        if len(kpts) < 36:
            return None, None
        mw = md.get("model_width", 960.0)
        mh = md.get("model_height", 544.0)
        img, court = [], []
        for i in (3, 4, 5, 6, 7, 8):
            x, y, c = kpts[3 * i:3 * i + 3]
            if c < 0.25:
                continue
            img.append((x / mw, y / mh))
            court.append(_POSE_CANON[i] / np.array([COURT_W, COURT_H]))
        if not img:
            return None, None
        return np.array(img), np.array(court)

    PLAYER_HEIGHT_FT = 6.2

    def _players_scale_ratio(self, frame, prm, w, h):
        """Median ratio of detected player bbox heights to the height a
        ~6 ft person must subtend under the camera at each bbox's floor
        position. A wrong-scale fit (misclassified close-up) is off by 2-10x
        coherently — regardless of which evidence path produced it. None if
        too few usable players."""
        md = self._read_json(frame, self.player_metadata_key)
        if not md:
            return None
        h8 = self._ptz_to_h8(tuple(prm), w, h)
        if h8 is None:
            return None
        mw_ = md.get("model_width", 960.0)
        mh_ = md.get("model_height", 544.0)
        ratios = []
        n_out = 0
        for det in md.get("detections", []):
            if det.get("label") not in ("Player", "Ref") or "xyxy" not in det:
                continue
            x1, y1, x2, y2 = det["xyxy"]
            foot = np.array([[(x1 + x2) * 0.5 / mw_, y2 / mh_]])
            c = _apply_h(h8, foot)[0]
            if not np.all(np.isfinite(c)):
                n_out += 1
                continue
            X, Y = c[0] * COURT_W, c[1] * COURT_H
            if not (-5.0 <= X <= COURT_W + 5.0 and -5.0 <= Y <= COURT_H + 8.0):
                # people standing nowhere near the court under this camera —
                # itself evidence the camera is wrong (do not silently skip)
                n_out += 1
                continue
            pr = self._ptz_project(prm, np.array(
                [[X, Y, 0.0], [X, Y, self.PLAYER_HEIGHT_FT]]), w, h)
            if not np.isfinite(pr).all():
                continue
            exp_h = pr[0, 1] - pr[1, 1]
            det_h = (y2 - y1) / mh_
            if exp_h > 0.02 and det_h > 0.02:
                ratios.append(det_h / exp_h)
        if n_out >= 3 and n_out > len(ratios):
            return 0.0  # sentinel: majority of people land off-court
        if len(ratios) < 3:
            return None
        return float(np.median(ratios))

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
        binm = m >= self.mask_threshold
        if not binm.any():
            return m
        lbl, n = ndimage.label(binm)
        if n > 1:
            sizes = ndimage.sum(binm, lbl, index=np.arange(1, n + 1))
            binm = lbl == (int(np.argmax(sizes)) + 1)
        filled = ndimage.binary_fill_holes(binm)
        out = m.copy()
        # drop above-threshold specks / extra components outside the kept blob
        out[(m >= self.mask_threshold) & ~filled] = np.float32(0.0)
        # raise filled-in court-bleed holes above threshold; sub-threshold
        # anti-aliased boundary pixels are left untouched so find_contours
        # keeps its sub-pixel 0.5-crossing accuracy at the painted edge.
        out[filled & (m < self.mask_threshold)] = np.float32(self.mask_threshold + 0.25)
        return out

    def _reconcile_spillover(self, tpl, court, prior_h8, w, h):
        """Move court-class activation that falls inside the reprojected 3-pt
        region back into the 3-pt mask.

        Inside the painted arc only the 3-pt region is physically possible, so
        any `basketball-court` evidence there is segmentation spillover. Project
        each mask pixel forward through the prior homography (source-norm ->
        court-feet) and test analytic membership with `_inside_tpl_region` (the
        same test the templates use). Court activation inside that region is
        reclassified: added to the 3-pt mask, removed from the court mask. This
        recovers 3-pt area the court class stole (clean_tpl_mask can only fill
        already-enclosed holes), rescuing frames that would otherwise collapse
        to region_small. Returns possibly-updated (tpl, court); a None prior or
        a degenerate projection is a no-op so the frame is never regressed.
        """
        if prior_h8 is None or tpl is None or court is None:
            return tpl, court
        mh, mw = tpl.shape
        if court.shape != tpl.shape:
            return tpl, court
        # normalized-source pixel grid (mask resolution), projected to court ft
        xn, _ = self._mask_idx_to_norm(np.arange(mw), np.zeros(mw), mw, mh)
        _, yn = self._mask_idx_to_norm(np.zeros(mh), np.arange(mh), mw, mh)
        gy, gx = np.meshgrid(yn, xn, indexing="ij")
        pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
        c = _apply_h(prior_h8, pix) * np.array([COURT_W, COURT_H])
        X = np.where(np.isfinite(c[:, 0]), c[:, 0], 1e6).reshape(mh, mw)
        Y = np.where(np.isfinite(c[:, 1]), c[:, 1], 1e6).reshape(mh, mw)
        inside = (self._inside_region_inset(X, Y, self.reconcile_margin_ft)
                  & (X >= 0) & (X <= COURT_W) & (Y >= 0) & (Y <= COURT_H))
        if not inside.any():
            return tpl, court
        # Spillover is court activation inside the arc WHERE the 3-pt class did
        # NOT fire. Both classes legitimately co-activate on the same floor, so
        # gating on tpl-absent isolates the pathological flood (3-pt collapsed,
        # court bled in) and makes this a strict no-op wherever the 3-pt mask is
        # healthy. That is why frames with good raw detections are left exactly
        # as detected -- only collapsed regions are rescued.
        spill = (inside & (court >= self.mask_threshold)
                 & (tpl < self.mask_threshold))
        if not spill.any():
            return tpl, court
        tpl = np.maximum(tpl, np.where(spill, court, np.float32(0.0)))
        court = np.where(spill, np.float32(0.0), court)
        return tpl, court

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
        binm = court >= self.mask_threshold
        if binm.mean() < 0.08:
            return None
        closed = binm.copy()
        closed[0, :] = True
        closed[-1, :] = True
        closed[:, 0] = True
        closed[:, -1] = True
        holes = ndimage.binary_fill_holes(closed) & ~closed & ~binm
        if not holes.any():
            return None
        lbl, n = ndimage.label(holes)
        out = np.zeros_like(court)
        found = False
        for i in range(1, n + 1):
            comp = lbl == i
            if comp.mean() < self.min_region_area:
                continue
            ring = ndimage.binary_dilation(comp) & ~comp
            nr = int(ring.sum())
            if nr == 0 or float((ring & binm).sum()) / nr < 0.5:
                continue  # bounded by border/crowd, not by court
            out[comp] = np.float32(self.mask_threshold + 0.25)
            found = True
        if not found:
            return None
        return self._clean_region_mask(out)

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

    def _tpl_boundary_points(self, tpl, w, h):
        """Sub-pixel boundary of the 3-pt region in source px, border-clipped."""
        contours = find_contours(tpl, self.mask_threshold)
        if not contours:
            return None
        contours.sort(key=len, reverse=True)
        if len(contours[0]) < 40:
            return None
        mh, mw = tpl.shape
        all_pts = []
        for c in contours:
            if len(c) < 20:
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
