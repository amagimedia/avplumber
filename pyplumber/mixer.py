"""Reusable avplumber video mixer graph builder.

Mirrors the graph structure from examples/mixer.avplumber and supports the full
MixerOrchestrator feature set: cut, crossfade, and media wipe transitions.

Typical usage
-------------
    from pyplumber import AVPlumber
    from pyplumber.mixer import MixerGraphBuilder
    from pyplumber.node import InputRec, Demux, DecVideo, Realtime, ForceFPS

    avp = AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 3)

    # Build input decode chains externally (the caller owns input groups)
    for i, url in enumerate(urls):
        avp.addNode(InputRec({"url": url, "dst": f"in{i}_pkt", "group": f"input_{i}", ...}))
        ...
        avp.addNode(ForceFPS({"fps": "30/1", "src": f"cam{i}_rt", "dst": f"cam{i}_fps",
                              "group": f"input_{i}"}))

    mx = MixerGraphBuilder(avp, name="mixer", canvas=(1920, 1080), fps=(30, 1),
                           hwaccel="@gpu", timeline="mixer_tl", enable_wipe=True)
    mx.add_source("cam0", pre_otm_edge="cam0_fps", input_group="input_0")
    mx.add_source("cam1", pre_otm_edge="cam1_fps", input_group="input_1")
    mx.add_scene("fullcam0", {"cam0": {"graph": "scale_cuda=w=1920:h=1080", "dst_x": 0, "dst_y": 0}})
    mx.add_scene("pip", {
        "cam0": {"graph": "scale_cuda=w=1920:h=1080", "dst_x": 0, "dst_y": 0},
        "cam1": {"graph": "scale_cuda=w=640:h=360", "dst_x": 1280, "dst_y": 720},
    })
    mx.set_initial_scene("fullcam0", slot="A")
    out_edge = mx.build()   # returns the name of the final output video edge

    # Add encode / mux / output after build()...

    # Start groups (input groups are the caller's responsibility)
    avp.group("input_0").startNodes()
    avp.group("input_1").startNodes()
    mx.start_groups()
    avp.group("output").startNodes()

    # Runtime control
    mx.cut("pip")
    mx.fade("fullcam0", duration_sec=2.0)
    mx.wipe("pip", wipe_file="/path/to/wipe.mov")
    print(mx.current_scene, mx.scenes())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .node import (
    CudaRectOverlay,
    FilterVideo,
    ForceFPS,
    InputRec,
    Demux,
    DecVideo,
    OneToMany,
    Realtime,
    SourceSwitcher,
)


@dataclass
class MixerSource:
    name: str
    pre_otm_edge: str
    input_group: str
    audio_edge: Optional[str] = None


@dataclass
class MixerScene:
    name: str
    # source_name -> {"graph": ..., "dst_x": ..., "dst_y": ..., ...}
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MixerGraphBuilder:
    """Build and operate an avplumber 2-slot video mixer (MixerOrchestrator).

    The builder owns all mixer-internal nodes.  The caller owns input decode
    chains, output encode/mux chains, and audio routing.  Audio edges are
    recorded in add_source() metadata for the caller's convenience but are
    not wired by this class.

    Node naming convention: every internal node or edge is prefixed with
    ``<name>_`` to allow multiple mixer instances in the same process.
    """

    def __init__(
        self,
        avp,
        name: str = "mixer",
        canvas: Tuple[int, int] = (1920, 1080),
        fps: Tuple[int, int] = (30, 1),
        hwaccel: str = "@gpu",
        timeline: Optional[str] = None,
        enable_wipe: bool = True,
    ):
        self.avp = avp
        self.name = name
        self.canvas_w, self.canvas_h = canvas
        self.fps_num, self.fps_den = fps
        self.hwaccel = hwaccel
        self.timeline = timeline or f"{name}_tl"
        self.enable_wipe = enable_wipe

        self._sources: List[MixerSource] = []
        self._source_index: Dict[str, int] = {}
        self._scenes: Dict[str, MixerScene] = {}
        self._initial_pgm_scene: Optional[str] = None
        self._initial_pgm_slot: str = "A"
        self._current_pgm: Optional[str] = None
        self._built = False

    # ------------------------------------------------------------------
    # Graph construction API
    # ------------------------------------------------------------------

    def add_source(
        self,
        name: str,
        pre_otm_edge: str,
        input_group: str,
        audio_edge: Optional[str] = None,
    ) -> "MixerGraphBuilder":
        """Register one camera source.

        The source's one_to_many and per-slot crop-scale nodes will be
        created in *input_group* during build() so that they restart
        together with the input decode chain.

        Parameters
        ----------
        name:
            Logical source identifier (must be unique).
        pre_otm_edge:
            Name of the avplumber edge that feeds into this source's
            one_to_many (typically the force_fps output of the input chain).
        input_group:
            Avplumber group that owns this source's OTM and crop-scale nodes.
        audio_edge:
            Optional name of the audio edge associated with this source.
            Not wired by the builder; stored as metadata for the caller.
        """
        if self._built:
            raise RuntimeError("Cannot add sources after build()")
        if name in self._source_index:
            raise ValueError(f"Source '{name}' already registered")
        idx = len(self._sources)
        self._sources.append(MixerSource(name, pre_otm_edge, input_group, audio_edge))
        self._source_index[name] = idx
        return self

    def add_scene(
        self,
        name: str,
        sources: Dict[str, Dict[str, Any]],
    ) -> "MixerGraphBuilder":
        """Define a named scene.

        Parameters
        ----------
        name:
            Scene identifier (must be unique).
        sources:
            Mapping from logical source name to a dict with at minimum
            ``graph`` (the FFmpeg filter chain for that camera's
            crop/scale filter_video) and optional compositor layer
            keys ``dst_x``, ``dst_y``, etc.
        """
        if self._built:
            raise RuntimeError("Cannot add scenes after build()")
        unknown = [s for s in sources if s not in self._source_index]
        if unknown:
            raise ValueError(f"Scene '{name}' references unknown source(s): {unknown}")
        self._scenes[name] = MixerScene(name=name, sources=sources)
        return self

    def set_initial_scene(self, scene_name: str, slot: str = "A") -> "MixerGraphBuilder":
        """Declare which scene starts on PGM.

        Must be called before build().  The initial compositor states
        (active_inputs bitmask, one_to_many outputs bitmask) will be
        derived from this scene definition.
        """
        if self._built:
            raise RuntimeError("Cannot set initial scene after build()")
        if slot not in ("A", "B"):
            raise ValueError("slot must be 'A' or 'B'")
        self._initial_pgm_scene = scene_name
        self._initial_pgm_slot = slot
        return self

    def build(self) -> str:
        """Materialize all mixer nodes and emit mixer.init/source/scene.

        Returns the name of the final output video edge (``<name>_final_out``).
        """
        if self._built:
            raise RuntimeError("build() called twice")
        if not self._sources:
            raise RuntimeError("No sources registered")
        if not self._scenes:
            raise RuntimeError("No scenes registered")
        if self._initial_pgm_scene is None:
            raise RuntimeError("set_initial_scene() not called")
        if self._initial_pgm_scene not in self._scenes:
            raise ValueError(
                f"Initial scene '{self._initial_pgm_scene}' not found in registered scenes"
            )

        self._build_per_source_nodes()
        self._build_compositors()
        self._build_output_path()
        if self.enable_wipe:
            self._build_wipe_subgraph()
        self._emit_mixer_commands()

        self._built = True
        self._current_pgm = self._initial_pgm_scene
        return self._e("final_out")

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------

    def cut(self, scene: str, start_pts_ms: int = -1) -> None:
        """Hard cut to *scene*."""
        cmd = {"mixer": self.name, "scene": scene}
        if start_pts_ms >= 0:
            cmd["start_pts_ms"] = start_pts_ms
        self.avp.executeCommandsFromString(f"mixer.cut {json.dumps(cmd)}")
        self._current_pgm = scene

    def fade(self, scene: str, duration_sec: float = 1.0, start_pts_ms: int = -1) -> None:
        """Crossfade to *scene* over *duration_sec* seconds."""
        cmd = {"mixer": self.name, "scene": scene, "duration_sec": duration_sec}
        if start_pts_ms >= 0:
            cmd["start_pts_ms"] = start_pts_ms
        self.avp.executeCommandsFromString(f"mixer.fade {json.dumps(cmd)}")
        self._current_pgm = scene

    def wipe(
        self,
        scene: str,
        wipe_file: str,
        duration_sec: Optional[float] = None,
        start_pts_ms: int = -1,
    ) -> None:
        """Media wipe to *scene* using *wipe_file* (must have alpha channel)."""
        if not self.enable_wipe:
            raise RuntimeError("Wipe subgraph not enabled (pass enable_wipe=True)")
        cmd: Dict[str, Any] = {"mixer": self.name, "scene": scene, "wipe_file": wipe_file}
        if duration_sec is not None:
            cmd["duration_sec"] = duration_sec
        if start_pts_ms >= 0:
            cmd["start_pts_ms"] = start_pts_ms
        self.avp.executeCommandsFromString(f"mixer.wipe {json.dumps(cmd)}")
        self._current_pgm = scene

    def scenes(self) -> List[str]:
        """Return the sorted list of registered scene names."""
        return sorted(self._scenes)

    @property
    def current_scene(self) -> Optional[str]:
        """Last scene requested (local tracking; not polled from the mixer)."""
        return self._current_pgm

    def start_groups(self) -> None:
        """Start the mixer's internal compositor and output groups.

        Input groups (which host per-source OTMs and crop-scale chains) are
        NOT started here; the caller is responsible for those because they
        share a lifecycle with the input decode chains.
        """
        for g in [f"{self.name}_a", f"{self.name}_b", self.name]:
            self.avp.group(g).startNodes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n(self, local: str) -> str:
        """Prefix a local node name with the mixer instance name."""
        return f"{self.name}_{local}"

    def _e(self, local: str) -> str:
        """Prefix a local edge name with the mixer instance name."""
        return f"{self.name}_{local}"

    def _fps_str(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"

    def _initial_scene_def(self) -> MixerScene:
        return self._scenes[self._initial_pgm_scene]

    def _active_inputs_mask(self, scene: MixerScene) -> int:
        """Compute the cuda_rect_overlay active_inputs bitmask for a scene."""
        mask = 0
        for src_name in scene.sources:
            idx = self._source_index[src_name]
            mask |= 1 << idx
        return mask

    def _build_per_source_nodes(self) -> None:
        """Create one_to_many + per-slot filter_video nodes for every source."""
        initial_scene = self._initial_scene_def()
        pgm_slot_bit = 0 if self._initial_pgm_slot == "A" else 1

        for src in self._sources:
            is_in_initial = src.name in initial_scene.sources
            # Slot A = bit 0, slot B = bit 1.
            # Initial PGM is on the configured slot; sources in the scene
            # feed only the PGM slot.
            outputs_init = (1 << pgm_slot_bit) if is_in_initial else 0

            self.avp.addNode(OneToMany({
                "type": "one_to_many",
                "name": self._n(f"otm_{src.name}"),
                "src": src.pre_otm_edge,
                "dst": [self._e(f"{src.name}_a"), self._e(f"{src.name}_b")],
                "outputs": outputs_init,
                "timeline": self.timeline,
                "group": src.input_group,
            }))

            # Default scale: fit to canvas.  MixerOrchestrator rewrites the
            # graph string on every scene switch via node.param.set + auto_restart.
            default_graph = (
                f"scale_cuda=w={self.canvas_w}:h={self.canvas_h}:interp_algo=lanczos"
            )

            self.avp.addNode(FilterVideo({
                "name": self._n(f"cs_{src.name}_a"),
                "src": self._e(f"{src.name}_a"),
                "dst": self._e(f"{src.name}_scaled_a"),
                "graph": default_graph,
                "hwaccel": self.hwaccel,
                "group": src.input_group,
                "auto_restart": "on",
            }))

            self.avp.addNode(FilterVideo({
                "name": self._n(f"cs_{src.name}_b"),
                "src": self._e(f"{src.name}_b"),
                "dst": self._e(f"{src.name}_scaled_b"),
                "graph": default_graph,
                "hwaccel": self.hwaccel,
                "group": src.input_group,
                "auto_restart": "on",
            }))

    def _build_compositors(self) -> None:
        """Create slot-A compositor, slot-B compositor, and their ancillary nodes."""
        initial_scene = self._initial_scene_def()
        pgm_is_a = self._initial_pgm_slot == "A"
        fps_str = self._fps_str()

        srcs_a = [self._e(f"{s.name}_scaled_a") for s in self._sources]
        srcs_b = [self._e(f"{s.name}_scaled_b") for s in self._sources]
        n = len(self._sources)

        active_pgm = self._active_inputs_mask(initial_scene)
        active_a = active_pgm if pgm_is_a else 0
        active_b = 0 if pgm_is_a else active_pgm

        default_layers = [{"dst_x": 0, "dst_y": 0}] * n

        self.avp.addNode(CudaRectOverlay({
            "name": self._n("comp_a"),
            "src": srcs_a,
            "dst": self._e("scene_a_out"),
            "hwaccel": self.hwaccel,
            "width": self.canvas_w,
            "height": self.canvas_h,
            "sw_format": "nv12",
            "layers": default_layers,
            "active_inputs": active_a,
            "timeline": self.timeline,
            "group": f"{self.name}_a",
        }))
        self.avp.addNode(ForceFPS({
            "name": self._n("norm_a"),
            "fps": fps_str,
            "src": self._e("scene_a_out"),
            "dst": self._e("scene_a_norm"),
            "group": f"{self.name}_a",
        }))
        self.avp.addNode(OneToMany({
            "name": self._n("otm_scene_a"),
            "src": self._e("scene_a_norm"),
            "dst": [self._e("scA_direct"), self._e("scA_trans")],
            "outputs": 1 if pgm_is_a else 0,
            "timeline": self.timeline,
            "group": f"{self.name}_a",
        }))

        self.avp.addNode(CudaRectOverlay({
            "name": self._n("comp_b"),
            "src": srcs_b,
            "dst": self._e("scene_b_out"),
            "hwaccel": self.hwaccel,
            "width": self.canvas_w,
            "height": self.canvas_h,
            "sw_format": "nv12",
            "layers": default_layers,
            "active_inputs": active_b,
            "timeline": self.timeline,
            "group": f"{self.name}_b",
        }))
        self.avp.addNode(ForceFPS({
            "name": self._n("norm_b"),
            "fps": fps_str,
            "src": self._e("scene_b_out"),
            "dst": self._e("scene_b_norm"),
            "group": f"{self.name}_b",
        }))
        self.avp.addNode(OneToMany({
            "name": self._n("otm_scene_b"),
            "src": self._e("scene_b_norm"),
            "dst": [self._e("scB_direct"), self._e("scB_trans")],
            "outputs": 0 if pgm_is_a else 1,
            "timeline": self.timeline,
            "group": f"{self.name}_b",
        }))

    def _build_output_path(self) -> None:
        """Create the output selector and wipe routing nodes."""
        fps_str = self._fps_str()
        pgm_is_a = self._initial_pgm_slot == "A"
        initial_active = 0 if pgm_is_a else 1

        self.avp.addNode(SourceSwitcher({
            "name": self._n("out_sel"),
            "src": [
                self._e("scA_direct"),
                self._e("scB_direct"),
                self._e("trans_out"),
            ],
            "dst": self._e("mixer_out"),
            "active": initial_active,
            "timeline": self.timeline,
            "group": self.name,
        }))
        self.avp.addNode(ForceFPS({
            "name": self._n("wipe_base_fps"),
            "fps": fps_str,
            "src": self._e("mixer_out"),
            "dst": self._e("final_wipe_pre"),
            "group": self.name,
        }))
        self.avp.addNode(OneToMany({
            "name": self._n("otm_final"),
            "src": self._e("final_wipe_pre"),
            "dst": [self._e("final_direct"), self._e("final_wipe_in")],
            "outputs": 1,
            "timeline": self.timeline,
            "group": self.name,
        }))
        self.avp.addNode(SourceSwitcher({
            "name": self._n("wipe_sel"),
            "src": [self._e("final_direct"), self._e("wipe_overlay_out")],
            "dst": self._e("final_out"),
            "active": 0,
            "timeline": self.timeline,
            "group": self.name,
        }))

    def _build_wipe_subgraph(self) -> None:
        """Create the pre-declared wipe subgraph (not started; orchestrator manages it)."""
        W, H = self.canvas_w, self.canvas_h
        fps_str = self._fps_str()
        wipe_group = f"{self.name}_wipe"

        self.avp.addNode(InputRec({
            "name": self._n("wipe_input"),
            "url": "",
            "loop": False,
            "dst": self._e("wipe_raw_pkt"),
            "group": wipe_group,
        }))
        self.avp.addNode(Demux({
            "name": self._n("wipe_demux"),
            "src": self._e("wipe_raw_pkt"),
            "routing": {"v:0": self._e("wipe_v_pkt")},
            "group": wipe_group,
        }))
        self.avp.addNode(DecVideo({
            "name": self._n("wipe_dec"),
            "src": self._e("wipe_v_pkt"),
            "dst": self._e("wipe_dec_out"),
            "pixel_format": "?cuda",
            "hwaccel": self.hwaccel,
            "group": wipe_group,
        }))
        # Scale wipe to canvas size, convert to YUVA420P for alpha support.
        # scale_cuda lacks YUVA support; use software scale + hwupload.
        self.avp.addNode(FilterVideo({
            "name": self._n("wipe_fmt"),
            "src": self._e("wipe_dec_out"),
            "dst": self._e("wipe_fmt_out"),
            "graph": f"format=yuva420p,scale={W}:{H}:flags=lanczos,hwupload_cuda",
            "hwaccel": self.hwaccel,
            "group": wipe_group,
        }))
        self.avp.addNode(Realtime({
            "name": self._n("wipe_rt"),
            "src": self._e("wipe_fmt_out"),
            "dst": self._e("wipe_rt_out"),
            "set_pts": True,
            "group": wipe_group,
        }))
        self.avp.addNode(ForceFPS({
            "name": self._n("wipe_rt_fps"),
            "fps": fps_str,
            "src": self._e("wipe_rt_out"),
            "dst": self._e("wipe_rt_fps_out"),
            "group": wipe_group,
        }))
        # overlay_many_cuda: convert NV12 main to YUV420P, blend YUVA wipe,
        # convert back to NV12 (matches assume_video_format downstream).
        self.avp.addNode(FilterVideo({
            "name": self._n("wipe_overlay"),
            "src": [self._e("final_wipe_in"), self._e("wipe_rt_fps_out")],
            "dst": self._e("wipe_overlay_out"),
            "graph": (
                "[in0]scale_cuda=format=yuv420p[main];"
                " [main][in1]overlay_many_cuda=inputs=2[blended];"
                " [blended]scale_cuda=format=nv12"
            ),
            "hwaccel": self.hwaccel,
            "group": wipe_group,
        }))

    def _emit_mixer_commands(self) -> None:
        """Issue mixer.init, mixer.source, and mixer.scene commands."""
        wipe_group = f"{self.name}_wipe"
        wipe_flush_edges = [
            self._e("wipe_raw_pkt"),
            self._e("wipe_v_pkt"),
            self._e("wipe_dec_out"),
            self._e("wipe_fmt_out"),
            self._e("wipe_rt_out"),
            self._e("wipe_rt_fps_out"),
            self._e("final_wipe_pre"),
            self._e("final_wipe_in"),
            self._e("wipe_overlay_out"),
        ]

        init_cfg: Dict[str, Any] = {
            "timeline": self.timeline,
            "hwaccel": self.hwaccel,
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "source_switcher": self._n("out_sel"),
            "initial_pgm_slot": self._initial_pgm_slot,
            "initial_pgm_scene": self._initial_pgm_scene,
            "wipe_otm": self._n("otm_final"),
            "wipe_base_fps": self._n("wipe_base_fps"),
            "wipe_selector": self._n("wipe_sel"),
            "slot_a": {
                "compositor": self._n("comp_a"),
                "norm_ts": self._n("norm_a"),
                "post_otm": self._n("otm_scene_a"),
            },
            "slot_b": {
                "compositor": self._n("comp_b"),
                "norm_ts": self._n("norm_b"),
                "post_otm": self._n("otm_scene_b"),
            },
        }

        if self.enable_wipe:
            init_cfg.update({
                "wipe_group": wipe_group,
                "wipe_input_node": self._n("wipe_input"),
                "wipe_tail_edge": self._e("wipe_rt_fps_out"),
                "wipe_flush_edges": wipe_flush_edges,
            })

        lines = [f"mixer.init {self.name} {json.dumps(init_cfg)}"]

        for idx, src in enumerate(self._sources):
            lines.append(
                f"mixer.source {self.name} {src.name}"
                f" {self._n('otm_' + src.name)} {idx}"
                f" {self._n('cs_' + src.name + '_a')}"
                f" {self._n('cs_' + src.name + '_b')}"
            )

        for scene_name, scene in self._scenes.items():
            scene_def = {"sources": scene.sources}
            lines.append(f"mixer.scene {self.name} {scene_name} {json.dumps(scene_def)}")

        self.avp.executeCommandsFromString("\n".join(lines))
