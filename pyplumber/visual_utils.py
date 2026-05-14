NOPTS = -9223372036854775808


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def box_width(box: list[float]) -> float:
    return max(0.0, box[2] - box[0])


def box_height(box: list[float]) -> float:
    return max(0.0, box[3] - box[1])


def box_area(box: list[float]) -> float:
    return box_width(box) * box_height(box)


def box_contains(box: list[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def intersects_region(box: list[float], region: dict) -> bool:
    rx1 = float(region.get("x1", float("-inf")))
    ry1 = float(region.get("y1", float("-inf")))
    rx2 = float(region.get("x2", float("inf")))
    ry2 = float(region.get("y2", float("inf")))
    cx, cy = box_center(box)
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def relative_box(box: list[float], face_box: list[float]) -> list[float]:
    face_w = max(1.0, box_width(face_box))
    face_h = max(1.0, box_height(face_box))
    return [
        (box[0] - face_box[0]) / face_w,
        (box[1] - face_box[1]) / face_h,
        (box[2] - face_box[0]) / face_w,
        (box[3] - face_box[1]) / face_h,
    ]


def absolute_box(rel: list[float], face_box: list[float]) -> list[float]:
    face_w = max(1.0, box_width(face_box))
    face_h = max(1.0, box_height(face_box))
    return [
        face_box[0] + rel[0] * face_w,
        face_box[1] + rel[1] * face_h,
        face_box[0] + rel[2] * face_w,
        face_box[1] + rel[3] * face_h,
    ]


def clamp_box_to_face(box: list[float], face_box: list[float]) -> list[float]:
    return [
        clamp(box[0], face_box[0], face_box[2]),
        clamp(box[1], face_box[1], face_box[3]),
        clamp(box[2], face_box[0], face_box[2]),
        clamp(box[3], face_box[1], face_box[3]),
    ]


def clamp_relative_box(rel: list[float]) -> list[float]:
    x1 = clamp(rel[0], 0.0, 1.0)
    y1 = clamp(rel[1], 0.0, 1.0)
    x2 = clamp(rel[2], 0.0, 1.0)
    y2 = clamp(rel[3], 0.0, 1.0)
    if x2 <= x1:
        x2 = clamp(x1 + 0.01, 0.0, 1.0)
    if y2 <= y1:
        y2 = clamp(y1 + 0.01, 0.0, 1.0)
    return [x1, y1, x2, y2]


def blend_box(old: list[float], new: list[float], alpha: float) -> list[float]:
    alpha = clamp(alpha, 0.0, 1.0)
    return [(1.0 - alpha) * old[i] + alpha * new[i] for i in range(4)]


def timestamp_seconds(frame) -> float:
    pts = frame.pts
    if int(pts.timestamp) == NOPTS:
        return float("nan")
    tb = pts.timebase
    if not tb or int(tb.den) == 0:
        return float(int(pts.timestamp))
    return float(int(pts.timestamp)) * float(tb.num) / float(tb.den)


def valid_pts(frame) -> bool:
    try:
        pts = int(frame.pts.timestamp)
    except (AttributeError, TypeError, ValueError):
        return False
    return pts != NOPTS


def timestamp_ms_delta(frame, start_pts: int | None) -> float:
    if start_pts is None:
        return 0.0
    pts = frame.pts
    tb = pts.timebase
    if not tb or int(tb.den) == 0:
        return float(int(pts.timestamp) - start_pts)
    return 1000.0 * float(int(pts.timestamp) - start_pts) * float(tb.num) / float(tb.den)
