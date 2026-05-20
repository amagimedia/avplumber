"""Geometry-level preheated scene sources for the auto mixer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pyplumber.mixer import MixerGraphBuilder
from pyplumber.node import PreheatVideoRouter, Split

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
    PreheatTemplate("orig_stack", "orig", _stack_slots, _orig_stack_graph),
    PreheatTemplate("orig_pip_thumb", "orig", _pip_slots, _orig_pip_thumb_graph),
)


def _source_name(template_key: str, slot: int) -> str:
    return f"hot_{template_key}_{slot}"


def _router_name(input_kind: str) -> str:
    return f"preheat_{input_kind}_router"


def _router_output_edge(template_key: str, slot: int, mixer_slot: str) -> str:
    return f"hot_{template_key}_{slot}_route_{mixer_slot.lower()}"


@dataclass(frozen=True)
class PreheatedSceneSources:
    """Edges for regular mixer sources plus routed hot geometry slots."""

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

    def add_route(self, routes: dict[str, int], template_key: str, slot: int, active_input: int) -> None:
        routes[self.source(template_key, slot)] = active_input

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
    """Build routed raw feeds, then register one source per hot geometry slot."""
    n = len(face_edges)
    templates = _enabled_templates(n)
    slot_counts = {template.key: template.slots(n) for template in templates}
    template_by_key = {template.key: template for template in templates}

    face_mixer_edges: list[str] = []
    orig_mixer_edges: list[str] = []

    face_router_inputs = list(face_edges)
    orig_router_inputs = list(orig_edges)

    if include_regular_mixer_edges:
        face_router_inputs = []
        orig_router_inputs = []
        for i, (face_edge, orig_edge, input_group) in enumerate(zip(face_edges, orig_edges, input_groups)):
            face_mixer = f"v{i}_face_for_mixer"
            orig_mixer = f"v{i}_orig_for_mixer"
            face_router = f"v{i}_face_for_preheat_router"
            orig_router = f"v{i}_orig_for_preheat_router"
            face_mixer_edges.append(face_mixer)
            orig_mixer_edges.append(orig_mixer)
            face_router_inputs.append(face_router)
            orig_router_inputs.append(orig_router)
            avp.addNode(Split({
                "name": f"split_preheat_face_{i}",
                "src": face_edge,
                "dst": [face_mixer, face_router],
                "drop": True,
                "group": input_group,
                "auto_restart": "group",
            }))
            avp.addNode(Split({
                "name": f"split_preheat_orig_{i}",
                "src": orig_edge,
                "dst": [orig_mixer, orig_router],
                "drop": True,
                "group": input_group,
                "auto_restart": "group",
            }))

    router_outputs: dict[str, list[str]] = {"face": [], "orig": []}
    router_labels: dict[str, list[str]] = {"face": [], "orig": []}
    route_indices: dict[tuple[str, int, str], int] = {}

    for template in templates:
        outputs = router_outputs[template.input_kind]
        labels = router_labels[template.input_kind]
        for slot in range(slot_counts[template.key]):
            for mixer_slot in ("A", "B"):
                route_indices[(template.key, slot, mixer_slot)] = len(outputs)
                outputs.append(_router_output_edge(template.key, slot, mixer_slot))
                labels.append(f"{template.key}_{slot}_{mixer_slot}")

    if router_outputs["face"]:
        avp.addNode(PreheatVideoRouter({
            "name": _router_name("face"),
            "src": face_router_inputs,
            "dst": router_outputs["face"],
            "routes": [-1] * len(router_outputs["face"]),
            "labels": router_labels["face"],
            "timeline": mx.timeline,
            "group": PREHEATED_GROUP,
            "auto_restart": "group",
        }))
    if router_outputs["orig"]:
        avp.addNode(PreheatVideoRouter({
            "name": _router_name("orig"),
            "src": orig_router_inputs,
            "dst": router_outputs["orig"],
            "routes": [-1] * len(router_outputs["orig"]),
            "labels": router_labels["orig"],
            "timeline": mx.timeline,
            "group": PREHEATED_GROUP,
            "auto_restart": "group",
        }))

    for template in templates:
        router = _router_name(template.input_kind)
        for slot in range(slot_counts[template.key]):
            mx.add_routed_source(
                _source_name(template.key, slot),
                pre_filter_edge_a=_router_output_edge(template.key, slot, "A"),
                pre_filter_edge_b=_router_output_edge(template.key, slot, "B"),
                input_group=PREHEATED_GROUP,
                route_router=router,
                route_output_a=route_indices[(template.key, slot, "A")],
                route_output_b=route_indices[(template.key, slot, "B")],
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


def _define_full_face_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    graph = preheated.graph("face_full")
    source = preheated.source("face_full")
    for i in range(n):
        routes: dict[str, int] = {}
        preheated.add_route(routes, "face_full", 0, i)
        mx.add_scene(
            f"full_face_{i}",
            {source: {"graph": graph, "dst_x": 0, "dst_y": 0}},
            routes=routes,
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
        routes: dict[str, int] = {}
        preheated.add_route(routes, "face_square", 0, i)

        for k, j in enumerate(others):
            sources[preheated.source("face_conf_thumb", k)] = {
                "graph": thumb_graph,
                "dst_x": x_off + k * cell_w,
                "dst_y": y_off,
            }
            preheated.add_route(routes, "face_conf_thumb", k, j)
        mx.add_scene(f"videoconf_{i}", sources, routes=routes)


def _define_vstack3_scenes(mx: MixerGraphBuilder, n: int, preheated: PreheatedSceneSources) -> None:
    if n < 3:
        return
    graph = preheated.graph("orig_stack")
    tile_h = 608
    top = (CANVAS_H - 3 * tile_h) // 2
    for a, b, c in sampled_ordered_triples(n, SAMPLED_MANUAL_SCENE_COUNT, VSTACK3_SCENE_SAMPLE_SEED):
        cams = [a, b, c]
        sources = {}
        routes: dict[str, int] = {}
        for slot, cam in enumerate(cams):
            sources[preheated.source("orig_stack", slot)] = {
                "graph": graph,
                "dst_x": 0,
                "dst_y": top + slot * tile_h,
            }
            preheated.add_route(routes, "orig_stack", slot, cam)
        mx.add_scene(f"vstack3_{a}_{b}_{c}", sources, routes=routes)


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
        routes: dict[str, int] = {}
        preheated.add_route(routes, "face_full", 0, i)
        preheated.add_route(routes, "orig_pip_thumb", 0, j)
        mx.add_scene(
            f"pip_{i}_{j}",
            {
                face_source: {"graph": face_graph, "dst_x": 0, "dst_y": 0},
                thumb_source: {"graph": thumb_graph, "dst_x": pip_x, "dst_y": pip_y},
            },
            routes=routes,
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
        routes: dict[str, int] = {}
        for slot, cam in enumerate(cams):
            sources[preheated.source("orig_stack", slot)] = {
                "graph": graph,
                "dst_x": 0,
                "dst_y": gap + slot * tile_h,
            }
            preheated.add_route(routes, "orig_stack", slot, cam)
        mx.add_scene(f"vstack_{a}_{b}", sources, routes=routes)
