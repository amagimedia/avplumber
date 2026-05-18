"""Geometry-level preheated scene sources for the auto mixer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pyplumber.mixer import MixerGraphBuilder
from pyplumber.node import SourceSwitcher, Split

from .config import (
    CANVAS_H,
    CANVAS_W,
    FACE_CROP_W,
    PIP_SCENE_SAMPLE_SEED,
    SAMPLED_MANUAL_SCENE_COUNT,
    VSTACK2_SCENE_SAMPLE_SEED,
    VSTACK3_SCENE_SAMPLE_SEED,
)
from .scenes import sampled_ordered_pairs, sampled_ordered_triples


PREHEATED_GROUP = "preheated_scene_geometry"


def _face_full_graph() -> str:
    return f"scale_cuda=w={CANVAS_W}:h={CANVAS_H}:interp_algo=lanczos"


def _face_square_graph() -> str:
    return (
        f"crop_cuda=w={FACE_CROP_W}:h={FACE_CROP_W}:x=0:y=0,"
        f"scale_cuda=w={CANVAS_W}:h={CANVAS_W}:interp_algo=lanczos"
    )


def _face_conf_thumb_graph() -> str:
    cell_w, cell_h = _face_conf_thumb_size()
    return f"scale_cuda=w={cell_w}:h={cell_h}:interp_algo=lanczos"


def _face_conf_thumb_size() -> tuple[int, int]:
    n_oth = 5
    bot_h = CANVAS_H - CANVAS_W
    cell_w = (CANVAS_W // n_oth) & ~1
    cell_h = (cell_w * 16 // 9) & ~1
    if cell_h > bot_h:
        cell_h = bot_h & ~1
        cell_w = (cell_h * 9 // 16) & ~1
    return cell_w, cell_h


def _face_grid_graph() -> str:
    cell_w = CANVAS_W // 2
    cell_h = cell_w * 16 // 9
    return f"scale_cuda=w={cell_w}:h={cell_h}:interp_algo=lanczos"


def _orig_stack_graph() -> str:
    return f"scale_cuda=w={CANVAS_W}:h=608:interp_algo=lanczos"


def _orig_pip_thumb_graph() -> str:
    pip_w = CANVAS_W // 3
    pip_h = (pip_w * 9 // 16) & ~1
    return f"scale_cuda=w={pip_w}:h={pip_h}:interp_algo=lanczos"


def _one_if_any(n: int) -> int:
    return 1 if n >= 1 else 0


def _conference_thumb_slots(n: int) -> int:
    return min(max(n - 1, 0), 5)


def _grid_slots(n: int) -> int:
    return min(n, 4) if n >= 3 else 0


def _stack_slots(n: int) -> int:
    if n >= 3:
        return 3
    if n >= 2:
        return 2
    return 0


def _pip_slots(n: int) -> int:
    return 1 if n >= 2 else 0


@dataclass(frozen=True)
class PreheatTemplate:
    key: str
    input_kind: str
    slots: Callable[[int], int]
    graph: Callable[[], str]


# Central preheat configuration. Each entry is one unique filter geometry; the
# slot count is the number of concurrent instances a layout can display.
PREHEAT_TEMPLATES: tuple[PreheatTemplate, ...] = (
    PreheatTemplate("face_full", "face", _one_if_any, _face_full_graph),
    PreheatTemplate("face_square", "face", _one_if_any, _face_square_graph),
    PreheatTemplate("face_conf_thumb", "face", _conference_thumb_slots, _face_conf_thumb_graph),
    PreheatTemplate("face_grid", "face", _grid_slots, _face_grid_graph),
    PreheatTemplate("orig_stack", "orig", _stack_slots, _orig_stack_graph),
    PreheatTemplate("orig_pip_thumb", "orig", _pip_slots, _orig_pip_thumb_graph),
)


def _source_name(template_key: str, slot: int) -> str:
    return f"hot_{template_key}_{slot}"


def _selector_name(template_key: str, slot: int) -> str:
    return f"hot_{template_key}_{slot}_selector"


def _selected_edge(template_key: str, slot: int) -> str:
    return f"hot_{template_key}_{slot}_selected"


def _selector_input_edge(input_index: int, template_key: str, slot: int) -> str:
    return f"v{input_index}_{template_key}_{slot}_for_selector"


@dataclass(frozen=True)
class PreheatedSceneSources:
    """Edges for regular mixer sources plus shared geometry selectors."""

    face_mixer_edges: list[str]
    orig_mixer_edges: list[str]
    templates: dict[str, PreheatTemplate]
    slot_counts: dict[str, int]

    def face_edge_for_mixer(self, index: int) -> str:
        return self.face_mixer_edges[index]

    def orig_edge_for_mixer(self, index: int) -> str:
        return self.orig_mixer_edges[index]

    def source(self, template_key: str, slot: int = 0) -> str:
        self.require_slot(template_key, slot)
        return _source_name(template_key, slot)

    def graph(self, template_key: str) -> str:
        return self.templates[template_key].graph()

    def control(self, template_key: str, slot: int, active_input: int) -> dict:
        self.require_slot(template_key, slot)
        return {
            "node": _selector_name(template_key, slot),
            "key": "active",
            "value": active_input,
        }

    def require_slot(self, template_key: str, slot: int) -> None:
        if self.slot_counts.get(template_key, 0) <= slot:
            raise ValueError(f"Preheat template '{template_key}' slot {slot} is not available")

    def summary(self) -> str:
        parts = []
        for template in PREHEAT_TEMPLATES:
            count = self.slot_counts.get(template.key, 0)
            if count:
                parts.append(f"{template.key}x{count}")
        return ", ".join(parts)


def _enabled_templates(n: int) -> tuple[PreheatTemplate, ...]:
    return tuple(template for template in PREHEAT_TEMPLATES if template.slots(n) > 0)


def build_preheated_scene_sources(
    avp,
    mx: MixerGraphBuilder,
    *,
    face_edges: list[str],
    orig_edges: list[str],
    input_groups: list[str],
    include_regular_mixer_edges: bool = False,
) -> PreheatedSceneSources:
    """Build selected raw feeds, then register one source per hot geometry slot."""
    n = len(face_edges)
    templates = _enabled_templates(n)
    slot_counts = {template.key: template.slots(n) for template in templates}
    template_by_key = {template.key: template for template in templates}

    face_mixer_edges: list[str] = []
    orig_mixer_edges: list[str] = []

    for i, (face_edge, orig_edge, input_group) in enumerate(zip(face_edges, orig_edges, input_groups)):
        face_outputs = []
        orig_outputs = []

        if include_regular_mixer_edges:
            face_outputs.append(f"v{i}_face_for_mixer")
            orig_outputs.append(f"v{i}_orig_for_mixer")
            face_mixer_edges.append(face_outputs[0])
            orig_mixer_edges.append(orig_outputs[0])

        for template in templates:
            outputs = face_outputs if template.input_kind == "face" else orig_outputs
            for slot in range(slot_counts[template.key]):
                outputs.append(_selector_input_edge(i, template.key, slot))

        if face_outputs:
            avp.addNode(Split({
                "name": f"split_preheat_face_{i}",
                "src": face_edge,
                "dst": face_outputs,
                "drop": True,
                "group": input_group,
                "auto_restart": "group",
            }))
        if orig_outputs:
            avp.addNode(Split({
                "name": f"split_preheat_orig_{i}",
                "src": orig_edge,
                "dst": orig_outputs,
                "drop": True,
                "group": input_group,
                "auto_restart": "group",
            }))

    for template in templates:
        for slot in range(slot_counts[template.key]):
            selector_inputs = [
                _selector_input_edge(i, template.key, slot)
                for i in range(n)
            ]
            avp.addNode(SourceSwitcher({
                "name": _selector_name(template.key, slot),
                "src": selector_inputs,
                "dst": _selected_edge(template.key, slot),
                "active": 0,
                "timeline": mx.timeline,
                "group": PREHEATED_GROUP,
                "auto_restart": "group",
            }))
            mx.add_source(
                _source_name(template.key, slot),
                pre_otm_edge=_selected_edge(template.key, slot),
                input_group=PREHEATED_GROUP,
                default_graph=template.graph(),
            )

    return PreheatedSceneSources(
        face_mixer_edges=face_mixer_edges,
        orig_mixer_edges=orig_mixer_edges,
        templates=template_by_key,
        slot_counts=slot_counts,
    )


def define_preheated_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    """Register the auto mixer scene set using preheated geometry sources."""
    _define_full_face_scenes(mx, n, preheated)
    _define_videoconf_scenes(mx, n, preheated)
    _define_vstack3_scenes(mx, n, preheated)
    _define_pip_scenes(mx, n, preheated)
    _define_vstack2_scenes(mx, n, preheated)
    _define_multiviewer_scene(mx, n, preheated)


def _define_full_face_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    graph = preheated.graph("face_full")
    source = preheated.source("face_full")
    for i in range(n):
        mx.add_scene(
            f"full_face_{i}",
            {source: {"graph": graph, "dst_x": 0, "dst_y": 0}},
            controls=[preheated.control("face_full", 0, i)],
        )


def _define_videoconf_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    if n < 2:
        return
    bot_h = CANVAS_H - CANVAS_W
    thumb_graph = preheated.graph("face_conf_thumb")
    square_graph = preheated.graph("face_square")

    cell_w, cell_h = _face_conf_thumb_size()
    y_off = CANVAS_W + (bot_h - cell_h) // 2

    for i in range(n):
        others = [j for j in range(n) if j != i][:preheated.slot_counts["face_conf_thumb"]]
        x_off = (CANVAS_W - len(others) * cell_w) // 2
        sources = {
            preheated.source("face_square"): {
                "graph": square_graph,
                "dst_x": 0,
                "dst_y": 0,
            }
        }
        controls = [preheated.control("face_square", 0, i)]

        for k, j in enumerate(others):
            sources[preheated.source("face_conf_thumb", k)] = {
                "graph": thumb_graph,
                "dst_x": x_off + k * cell_w,
                "dst_y": y_off,
            }
            controls.append(preheated.control("face_conf_thumb", k, j))
        mx.add_scene(f"videoconf_{i}", sources, controls=controls)


def _define_vstack3_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    if n < 3:
        return
    graph = preheated.graph("orig_stack")
    tile_h = 608
    top = (CANVAS_H - 3 * tile_h) // 2
    for a, b, c in sampled_ordered_triples(n, SAMPLED_MANUAL_SCENE_COUNT, VSTACK3_SCENE_SAMPLE_SEED):
        cams = [a, b, c]
        sources = {}
        controls = []
        for slot, cam in enumerate(cams):
            sources[preheated.source("orig_stack", slot)] = {
                "graph": graph,
                "dst_x": 0,
                "dst_y": top + slot * tile_h,
            }
            controls.append(preheated.control("orig_stack", slot, cam))
        mx.add_scene(f"vstack3_{a}_{b}_{c}", sources, controls=controls)


def _define_pip_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    if n < 2:
        return
    face_source = preheated.source("face_full")
    face_graph = preheated.graph("face_full")
    thumb_source = preheated.source("orig_pip_thumb")
    thumb_graph = preheated.graph("orig_pip_thumb")
    pip_w = CANVAS_W // 3
    pip_x = CANVAS_W - pip_w - 16
    pip_y = 16

    for i, j in sampled_ordered_pairs(n, SAMPLED_MANUAL_SCENE_COUNT, PIP_SCENE_SAMPLE_SEED):
        mx.add_scene(
            f"pip_{i}_{j}",
            {
                face_source: {"graph": face_graph, "dst_x": 0, "dst_y": 0},
                thumb_source: {"graph": thumb_graph, "dst_x": pip_x, "dst_y": pip_y},
            },
            controls=[
                preheated.control("face_full", 0, i),
                preheated.control("orig_pip_thumb", 0, j),
            ],
        )


def _define_vstack2_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    if n < 2:
        return
    graph = preheated.graph("orig_stack")
    tile_h = 608
    gap = (CANVAS_H - 2 * tile_h) // 2
    for a, b in sampled_ordered_pairs(n, SAMPLED_MANUAL_SCENE_COUNT, VSTACK2_SCENE_SAMPLE_SEED):
        cams = [a, b]
        sources = {}
        controls = []
        for slot, cam in enumerate(cams):
            sources[preheated.source("orig_stack", slot)] = {
                "graph": graph,
                "dst_x": 0,
                "dst_y": gap + slot * tile_h,
            }
            controls.append(preheated.control("orig_stack", slot, cam))
        mx.add_scene(f"vstack_{a}_{b}", sources, controls=controls)


def _define_multiviewer_scene(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    if n < 3:
        return
    graph = preheated.graph("face_grid")
    cols = 2
    cell_w = CANVAS_W // cols
    cell_h = cell_w * 16 // 9
    grid_n = min(n, preheated.slot_counts["face_grid"])
    sources = {}
    controls = []
    for j in range(grid_n):
        row, col = j // cols, j % cols
        sources[preheated.source("face_grid", j)] = {
            "graph": graph,
            "dst_x": col * cell_w,
            "dst_y": row * cell_h,
        }
        controls.append(preheated.control("face_grid", j, j))
    mx.add_scene("multiviewer", sources, controls=controls)
