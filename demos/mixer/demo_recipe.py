"""Expand weighted source and layout choices into reproducible demo scenes."""

import math
import random


LAYOUTS = ("fullscreen", "grid_2", "grid_4", "grid_8", "grid_16", "grid_32", "grid_64",
           "pip", "random", "alpha_overlay")


def allocate(total, weights):
    """Largest remainder allocation; ties follow recipe order."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("total must be a positive integer")
    if not weights or any(isinstance(w, bool) or not isinstance(w, (int, float)) or
                          not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must be finite non-negative numbers")
    scale = max(weights)
    if not scale:
        raise ValueError("at least one weight must be positive")
    normalized = [w / scale for w in weights]
    quotas = [total * w / sum(normalized) for w in normalized]
    counts = [math.floor(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda i: -(quotas[i] - counts[i]))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def scenes(sources, width, height, count, weights, seed, alpha_background=None):
    unknown = set(weights) - set(LAYOUTS)
    if unknown:
        raise ValueError(f"unknown layouts: {sorted(unknown)}")
    counts = allocate(count, list(weights.values()))
    rng = random.Random(seed)
    ids = [s["id"] for s in sources]
    browsers = [s["id"] for s in sources if s["kind"] == "browser"]
    videos = [s["id"] for s in sources if s["kind"] != "browser"]
    if alpha_background is not None and alpha_background not in videos:
        raise ValueError("alpha_background must name an allocated video source")
    result = []

    def item(source, x=0, y=0, w=width, h=height, blend=False):
        return {"source": source, "dst": {"x": x, "y": y, "w": w, "h": h},
                "fit": "contain", **({"blend": True} if blend else {})}

    def random_box(source, minimum, maximum):
        fraction = rng.uniform(minimum, maximum)
        w, h = max(2, int(width * fraction) & ~1), max(2, int(height * fraction) & ~1)
        return item(source, rng.randrange((width - w) // 2 + 1) * 2,
                    rng.randrange((height - h) // 2 + 1) * 2, w, h)

    for layout, amount in zip(weights, counts):
        if amount and layout == "alpha_overlay" and not (browsers and videos):
            raise ValueError("alpha_overlay requires at least one allocated browser and one video source")
        for _ in range(amount):
            # Rotation visits the whole source pool even with only small layouts.
            offset = len(result) % len(ids)
            selected = ids[offset:] + ids[:offset]
            if layout == "fullscreen":
                items = [item(selected[0])]
            elif layout.startswith("grid_"):
                capacity = int(layout.removeprefix("grid_"))
                columns = {2: 2, 4: 2, 8: 4, 16: 4, 32: 8, 64: 8}[capacity]
                rows = capacity // columns
                if height > width:
                    columns, rows = rows, columns
                xs = [(width * c // columns) & ~1 for c in range(columns + 1)]
                ys = [(height * r // rows) & ~1 for r in range(rows + 1)]
                items = [item(sid, xs[i % columns], ys[i // columns],
                              xs[i % columns + 1] - xs[i % columns],
                              ys[i // columns + 1] - ys[i // columns])
                         for i, sid in enumerate(selected[:capacity])]
            elif layout == "alpha_overlay":
                # Cycle backgrounds so a small test covers SDR and HDR inputs.
                index = len(result)
                background = alpha_background or videos[index % len(videos)]
                items = [item(background), item(browsers[index % len(browsers)], blend=True)]
            else:
                # The background is always present; later entries draw on top.
                rng.shuffle(selected)
                maximum = min(len(selected), 4 if layout == "pip" else 8)
                n = rng.randint(min(2, maximum), maximum)
                bounds = (0.22, 0.36) if layout == "pip" else (0.25, 0.70)
                items = [item(selected[0])] + [random_box(sid, *bounds) for sid in selected[1:n]]
            result.append({"id": f"{layout}_{len(result):03d}", "items": items})
    return result
