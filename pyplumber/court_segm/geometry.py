"""Court model + physical PTZ camera geometry (basketball calibration).

Pure geometry shared by the calibration pipeline: court constants, the
3-pt-region model polyline, the homography helpers, and the analytic
camera model. Everything here is stateless (module functions or static-ish
mixin methods) — the natural C++/CUDA port unit for per-point math.
"""

import math

import numpy as np

COURT_W = 94.0
COURT_H = 50.0
HOOP_X = 5.25
ARC_R = 23.75
CORNER_LAT = 22.0
JUNCTION_X = HOOP_X + math.sqrt(ARC_R * ARC_R - CORNER_LAT * CORNER_LAT)  # ~14.19


# Canonical court positions of the 12-keypoint court pose model (ft).
_POSE_CANON = np.array([
    (0.0, 0.0), (0.0, 25.0), (0.0, 50.0),
    (23.5, 0.0), (23.5, 50.0),
    (47.0, 0.0), (47.0, 50.0),
    (70.5, 0.0), (70.5, 50.0),
    (94.0, 0.0), (94.0, 25.0), (94.0, 50.0),
])

# Boundary segment labels of the 3-pt region.
SEG_FAR_STRAIGHT = 0
SEG_ARC = 1
SEG_NEAR_STRAIGHT = 2
SEG_BASELINE = 3


def _build_model_polyline(step_ft=0.5):
    """Dense closed boundary of the left-side 3-pt region, court feet,
    plus a per-point segment label.

    Corner cycle (image far->near for a left hoop): (0,3) -> (14.19,3) ->
    arc -> (14.19,47) -> (0,47) -> baseline -> (0,3).
    """
    pts = []
    labels = []

    def seg(a, b, label):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step_ft))
        for i in range(n):
            t = i / n
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            labels.append(label)

    y_far = 25.0 - CORNER_LAT  # 3.0
    y_near = 25.0 + CORNER_LAT  # 47.0
    seg((0.0, y_far), (JUNCTION_X, y_far), SEG_FAR_STRAIGHT)
    theta = math.asin(CORNER_LAT / ARC_R)
    n_arc = max(8, int(2 * theta * ARC_R / step_ft))
    for i in range(n_arc + 1):
        t = -theta + 2 * theta * i / n_arc
        pts.append((HOOP_X + ARC_R * math.cos(t), 25.0 + ARC_R * math.sin(t)))
        labels.append(SEG_ARC)
    seg((JUNCTION_X, y_near), (0.0, y_near), SEG_NEAR_STRAIGHT)
    seg((0.0, y_near), (0.0, y_far), SEG_BASELINE)
    return np.array(pts), np.array(labels)


_MODEL_LEFT, _MODEL_LABELS = _build_model_polyline()


def _mirror_x(pts):
    out = pts.copy()
    out[:, 0] = COURT_W - out[:, 0]
    return out


def _apply_h(h8, pts_norm):
    """h8: 8-vector; pts_norm: (N,2) normalized source -> (N,2) normalized court."""
    x, y = pts_norm[:, 0], pts_norm[:, 1]
    den = h8[6] * x + h8[7] * y + 1.0
    den = np.where(np.abs(den) < 1e-9, np.nan, den)
    cx = (h8[0] * x + h8[1] * y + h8[2]) / den
    cy = (h8[3] * x + h8[4] * y + h8[5]) / den
    return np.stack([cx, cy], axis=1)


class GeometryMixin:
    """Analytic camera/court geometry for CourtCalibrationNode."""

    RIM_HEIGHT_FT = 10.0

    @staticmethod
    def _ptz_to_h8(params, w, h):
        """Physical camera -> source-norm->court-norm homography (8-vec).
        params: pan, tilt, F(focal/width), cx, cy, cz (ft). World: X court
        length, Y across court (camera side y>50), Z up. With 3 per-frame DOF
        the far-field warp of a free 8-DOF fit is impossible by construction."""
        th, ph, F, cx, cy, cz = params
        a = w / h
        fwd = np.array([math.sin(th) * math.cos(ph),
                        -math.cos(th) * math.cos(ph),
                        -math.sin(ph)])
        right = np.array([math.cos(th), math.sin(th), 0.0])
        up = np.cross(right, fwd)
        if up[2] < 0:
            up = -up
        dn = -up  # image y grows downward
        C = np.array([cx, cy, cz])
        B = np.array([
            [F * right[0], F * right[1], -F * np.dot(C, right)],
            [F * a * dn[0], F * a * dn[1], -F * a * np.dot(C, dn)],
            [fwd[0], fwd[1], -np.dot(C, fwd)],
        ])
        B[0] += 0.5 * B[2]
        B[1] += 0.5 * B[2]
        Hci = B @ np.diag([COURT_W, COURT_H, 1.0])  # court-norm -> img-norm
        try:
            Hic = np.linalg.inv(Hci)
        except np.linalg.LinAlgError:
            return None
        if abs(Hic[2, 2]) < 1e-12:
            return None
        Hic = Hic / Hic[2, 2]
        return np.array([Hic[0, 0], Hic[0, 1], Hic[0, 2],
                         Hic[1, 0], Hic[1, 1], Hic[1, 2],
                         Hic[2, 0], Hic[2, 1]])

    @staticmethod
    def _ptz_project(prm, pts3d, w, h):
        """Project 3D court points (ft, Z up) through the physical camera;
        returns (N,2) normalized image coords (NaN behind camera)."""
        th, ph, F, cx, cy, cz = prm
        a = w / h
        fwd = np.array([math.sin(th) * math.cos(ph),
                        -math.cos(th) * math.cos(ph),
                        -math.sin(ph)])
        right = np.array([math.cos(th), math.sin(th), 0.0])
        up = np.cross(right, fwd)
        if up[2] < 0:
            up = -up
        dn = -up
        d = np.asarray(pts3d, dtype=float) - np.array([cx, cy, cz])
        z = d @ fwd
        z = np.where(np.abs(z) < 1e-9, np.nan, z)
        u = F * (d @ right) / z + 0.5
        v = F * a * (d @ dn) / z + 0.5
        return np.stack([u, v], axis=1)

    @staticmethod
    def _inside_tpl_region(X, Y):
        left = (X >= 0) & (np.abs(Y - 25.0) <= CORNER_LAT) & (
            (X <= JUNCTION_X) | (np.hypot(X - HOOP_X, Y - 25.0) <= ARC_R))
        Xm = COURT_W - X
        right = (Xm >= 0) & (np.abs(Y - 25.0) <= CORNER_LAT) & (
            (Xm <= JUNCTION_X) | (np.hypot(Xm - HOOP_X, Y - 25.0) <= ARC_R))
        return left | right

    @staticmethod
    def _inside_region_inset(X, Y, margin):
        """`_inside_tpl_region` shrunk inward by `margin` feet from every
        painted boundary (baseline, corner straights, arc). Used by spillover
        reconciliation to reclaim only the deep interior: the band next to the
        lines is left to the genuine segmentation so the arc-fit boundary is
        never replaced by a prior-H-shaped edge. margin<=0 -> the full region.
        """
        if margin <= 0:
            return GeometryMixin._inside_tpl_region(X, Y)
        lat = max(0.0, CORNER_LAT - margin)   # inset corner straights
        rad = max(0.0, ARC_R - margin)        # inset arc
        # left half: x in [margin .. JUNCTION_X] (baseline inset by margin) or
        # within the inset arc; bounded laterally by the inset corner straights.
        left = (X >= margin) & (np.abs(Y - 25.0) <= lat) & (
            (X <= JUNCTION_X) | (np.hypot(X - HOOP_X, Y - 25.0) <= rad))
        Xm = COURT_W - X
        right = (Xm >= margin) & (np.abs(Y - 25.0) <= lat) & (
            (Xm <= JUNCTION_X) | (np.hypot(Xm - HOOP_X, Y - 25.0) <= rad))
        return left | right

    def _rim_px_norm(self, h8, hoop_on_left, prm):
        """Image position (normalized) where the rim should appear for this
        camera: the analytic rim ground point (see _hoop_error_ptz_ft) mapped
        back through the ground homography."""
        cz = prm[5]
        if cz <= self.RIM_HEIGHT_FT + 1.0:
            return None
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        f = cz / (cz - self.RIM_HEIGHT_FT)
        ex = (prm[3] + (hx - prm[3]) * f) / COURT_W
        ey = (prm[4] + (25.0 - prm[4]) * f) / COURT_H
        hm = np.array([[h8[0], h8[1], h8[2]],
                       [h8[3], h8[4], h8[5]],
                       [h8[6], h8[7], 1.0]])
        try:
            hi = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            return None
        q = hi @ np.array([ex, ey, 1.0])
        if abs(q[2]) < 1e-9:
            return None
        return np.array([q[0] / q[2], q[1] / q[2]])

    def _quad_sane(self, h8, w, h):
        """Project the court corners to the image; a real broadcast view maps
        them to a large convex quad. Collapsed/degenerate fits cannot."""
        try:
            hm = np.array([[h8[0], h8[1], h8[2]],
                           [h8[3], h8[4], h8[5]],
                           [h8[6], h8[7], 1.0]])
            hi = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            return False
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        q = (hi @ np.hstack([corners, np.ones((4, 1))]).T).T
        den = q[:, 2]
        if np.any(np.abs(den) < 1e-9) or not np.all(np.isfinite(q)):
            return False
        xy = q[:, :2] / den[:, None] * np.array([w, h])
        # shoelace area, signed consistently => convex-ish and big enough
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area < 0.5 * w * h:
            return False
        cross = []
        for i in range(4):
            a, b, c = xy[i], xy[(i + 1) % 4], xy[(i + 2) % 4]
            cross.append(np.cross(b - a, c - b))
        return all(v > 0 for v in cross) or all(v < 0 for v in cross)

    @staticmethod
    def _curve_residuals_ft(proj, labels, hoop_on_left):
        """Per-point distance (ft) of projected court-feet points to their
        assigned analytic model curve; non-finite projections residual 50."""
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        base_x = 0.0 if hoop_on_left else COURT_W
        bad = ~np.isfinite(proj).all(axis=1)
        proj = np.where(bad[:, None], 0.0, proj)
        X, Y = proj[:, 0], proj[:, 1]
        r = np.zeros(len(proj))
        m = labels == SEG_FAR_STRAIGHT
        r[m] = Y[m] - (25.0 - CORNER_LAT)
        m = labels == SEG_NEAR_STRAIGHT
        r[m] = Y[m] - (25.0 + CORNER_LAT)
        m = labels == SEG_BASELINE
        r[m] = X[m] - base_x
        m = labels == SEG_ARC
        r[m] = np.hypot(X[m] - hx, Y[m] - 25.0) - ARC_R
        r[bad] = 50.0
        return r

    def _hoop_error_ptz_ft(self, h8, hoop_norm, hoop_on_left, rig):
        """Distance (ft) between the projected hoop detection and where the
        rim SHOULD land through a correct ground homography.

        The rim is RIM_HEIGHT_FT above the floor, so its image ray pierces
        the ground plane displaced away from the camera by a factor
        cz/(cz - rim) — ~38 ft for the broadcast rig, NOT "several ft". With
        the rig known, the expected point is analytic, making the hoop a
        strong absolute anchor against inflated/collapsed fits whose
        median curve residual still looks fine."""
        if hoop_norm is None:
            return None
        c = _apply_h(h8, np.array([hoop_norm]))[0]
        if not np.all(np.isfinite(c)):
            return float("inf")
        cx, cy, cz = rig
        if cz <= self.RIM_HEIGHT_FT + 1.0:
            return None
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        f = cz / (cz - self.RIM_HEIGHT_FT)
        ex = cx + (hx - cx) * f
        ey = cy + (25.0 - cy) * f
        return float(np.hypot(c[0] * COURT_W - ex, c[1] * COURT_H - ey))

    def _sideline_error_ft(self, h8, top_norm):
        """Median |Y| (ft) of projected court-mask top-edge points; the far
        sideline is y=0, so this measures global fit sanity independent of
        the 3-pt curves."""
        if top_norm is None or not len(top_norm):
            return None
        pr = _apply_h(h8, top_norm) * np.array([COURT_W, COURT_H])
        y = pr[:, 1]
        y = y[np.isfinite(y)]
        if len(y) < 8:
            return None
        return float(np.median(np.abs(y)))

    def _arc_coverage_deg(self, h8, tpl_pts, w, h, hoop_on_left):
        """Angular sweep (deg) of boundary points that land on the model arc.

        Projects the observed 3-pt boundary, keeps points within 3 ft of the
        arc radius and inside the corner-straight band, and measures the span
        of their angle about the hoop center. A baseline-only or near-side-only
        degenerate fit covers a small sweep even when its median residual is
        low; this catches that. Returns 0.0 if the arc is unsupported.
        """
        if tpl_pts is None or not len(tpl_pts):
            return 0.0
        scale = np.array([COURT_W, COURT_H])
        pts = tpl_pts[:: max(1, len(tpl_pts) // 300)] / np.array([w, h], dtype=float)
        proj = _apply_h(h8, pts) * scale
        ok = np.isfinite(proj).all(axis=1)
        proj = proj[ok]
        if len(proj) < 5:
            return 0.0
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        dx, dy = proj[:, 0] - hx, proj[:, 1] - 25.0
        if not hoop_on_left:
            dx = -dx  # mirror so the arc always spans dx>0, angles in (-90,90)
        r = np.hypot(dx, dy)
        # only the court-side half-plane: points at arc radius BEHIND the
        # hoop are not the arc, and their angles wrap +-180deg, inflating
        # the sweep of a degenerate fit to ~360
        on_arc = (dx > 0) & (np.abs(r - ARC_R) < 3.0) & (np.abs(dy) <= CORNER_LAT + 1.0)
        if on_arc.sum() < 5:
            return 0.0
        ang = np.degrees(np.arctan2(dy[on_arc], dx[on_arc]))
        return float(ang.max() - ang.min())
