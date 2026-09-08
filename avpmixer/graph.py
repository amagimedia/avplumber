"""Reusable AVPlumber video mixer graph builder.

Supports cuts, crossfades, and optional transparent media wipes.

Typical usage
-------------
    from pyplumber import AVPlumber
    from avpmixer import MixerGraphBuilder
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
from typing import Any, Dict, List, Optional, Tuple

from pyplumber.node import (
    InternalNode,
    ClipCache,
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


from . import clipcache
from .models import MixerScene, MixerSource
from .prewarm import TransitionPrewarm


class _SnapshotNode(InternalNode):
    TYPE = "mixer_snapshot"


class MixerGraphBuilder:
    """Build and operate an avplumber 2-slot video mixer (MixerOrchestrator).

    The builder owns all mixer-internal video nodes.  The caller owns input
    decode chains and output encode/mux chains.

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
        switch_margin_ms: int = 100,
        defer_initial_routes: bool = False,
        latency_ms: Optional[float] = None,
        defer_output: bool = False,
        keyframe_node: Optional[str] = None,
        cache_wipes_mb: Optional[float] = None,
    ):
        if switch_margin_ms < 0:
            raise ValueError("switch_margin_ms must be >= 0")
        self.avp = avp
        self.name = name
        self.canvas_w, self.canvas_h = canvas
        self.fps_num, self.fps_den = fps
        self.hwaccel = hwaccel
        self.timeline = timeline or f"{name}_tl"
        self.enable_wipe = enable_wipe
        self.switch_margin_ms = switch_margin_ms
        # Triggered when a transition reaches the output, so receivers do not wait
        # for the next periodic keyframe to see the new scene.
        self.keyframe_node = keyframe_node
        # When set, wipe clips are held decoded in GPU memory (avpmixer.clipcache)
        # and a take replays them instead of opening and decoding the file again.
        self.cache_wipes_mb = cache_wipes_mb
        self.defer_initial_routes = defer_initial_routes
        self.latency_ms = latency_ms
        self._output_started = not defer_output
        self._transition_prewarm = TransitionPrewarm(avp, self.name, self.timeline)

        self._sources: List[MixerSource] = []
        self._source_index: Dict[str, int] = {}
        self._scenes: Dict[str, MixerScene] = {}
        self._initial_pgm_scene: Optional[str] = None
        self._initial_pgm_slot: str = "A"
        self._routes_initialized = False
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
        default_graph: Optional[str] = None,
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
        default_graph:
            Initial crop/scale filter graph for this source's slot filters.
            An empty string bypasses slot filters; scenes then use compositor
            dst_w/dst_h and crop directly. Scene switches can still replace
            nonempty filters, but preheated geometry
            sources should start with their fixed graph to avoid a cold
            filter restart on the first take.
        """
        if self._built:
            raise RuntimeError("Cannot add sources after build()")
        if name in self._source_index:
            raise ValueError(f"Source '{name}' already registered")
        idx = len(self._sources)
        self._sources.append(MixerSource(name, pre_otm_edge, input_group, default_graph))
        self._source_index[name] = idx
        return self

    def add_routed_source(
        self,
        name: str,
        pre_filter_edge_a: str,
        pre_filter_edge_b: str,
        input_group: str,
        route_router: str,
        route_output_label_a: str,
        route_output_label_b: str,
        default_graph: Optional[str] = None,
    ) -> "MixerGraphBuilder":
        """Register a source whose slot filters are fed by a native preheat router."""
        if self._built:
            raise RuntimeError("Cannot add sources after build()")
        if name in self._source_index:
            raise ValueError(f"Source '{name}' already registered")
        idx = len(self._sources)
        self._sources.append(MixerSource(
            name=name,
            pre_otm_edge=None,
            input_group=input_group,
            default_graph=default_graph,
            pre_filter_edge_a=pre_filter_edge_a,
            pre_filter_edge_b=pre_filter_edge_b,
            route_router=route_router,
            route_output_label_a=route_output_label_a,
            route_output_label_b=route_output_label_b,
        ))
        self._source_index[name] = idx
        return self

    def add_scene(
        self,
        name: str,
        sources: Dict[str, Dict[str, Any]],
        controls: Optional[List[Dict[str, Any]]] = None,
        routes: Optional[Dict[str, int]] = None,
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
        return self.define_scene(name, sources, controls=controls, routes=routes)

    def define_scene(
        self,
        name: str,
        sources: Dict[str, Dict[str, Any]],
        controls: Optional[List[Dict[str, Any]]] = None,
        routes: Optional[Dict[str, int]] = None,
    ) -> "MixerGraphBuilder":
        """Define or replace a scene.

        Before build, this registers the scene for the initial mixer setup.
        After build, it also emits ``mixer.scene`` so runtime policies can
        reuse generic scene names with different source-slot assignments.
        """
        unknown = [s for s in sources if s not in self._source_index]
        if unknown:
            raise ValueError(f"Scene '{name}' references unknown source(s): {unknown}")
        routes = routes or {}
        route_unknown = [s for s in routes if s not in self._source_index]
        if route_unknown:
            raise ValueError(f"Scene '{name}' routes unknown source(s): {route_unknown}")
        scene = MixerScene(name=name, sources=sources, controls=controls or [], routes=routes)
        self._scenes[name] = scene
        if self._built:
            self.avp.executeCommandsFromString(self._scene_command(name, scene))
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

    def preview(self, scene: str) -> None:
        """Preload *scene* into the hidden PVW slot without taking it to program."""
        cmd = {"mixer": self.name, "scene": scene}
        self.avp.executeCommandsFromString(f"mixer.preview {json.dumps(cmd)}")

    def fade(
        self,
        scene: str,
        duration_sec: float = 1.0,
        start_pts_ms: int = -1,
    ) -> None:
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

    def warmup_wipe(self, wipe_file: str, timeout_ms: int = 30000) -> None:
        """Initialise the wipe chain on *wipe_file* once, invisibly, so the
        first real wipe does not pay for file open, decoder and GPU filter
        setup (PTX compilation included)."""
        if not self.enable_wipe:
            raise RuntimeError("Wipe subgraph not enabled (pass enable_wipe=True)")
        cmd = {"mixer": self.name, "wipe_file": wipe_file, "timeout_ms": timeout_ms}
        self.avp.executeCommandsFromString(f"mixer.wipe.warmup {json.dumps(cmd)}")

    def start_groups(self) -> None:
        """Start the mixer's internal compositor and output groups.

        Input groups (which host per-source OTMs and crop-scale chains) are
        NOT started here; the caller is responsible for those because they
        share a lifecycle with the input decode chains.
        """
        if not self._routes_initialized:
            raise RuntimeError(
                "initial mixer routes must be initialized before starting groups"
            )
        for g in [f"{self.name}_a", f"{self.name}_b", self.name]:
            self.avp.group(g).startNodes()

    def initialize_routes(self) -> None:
        """Publish the initial routed scene after optional graph preheating."""
        if self._routes_initialized:
            return
        self.avp.executeCommandsFromString(
            "mixer.init_routes " + json.dumps({"mixer": self.name})
        )
        self._routes_initialized = True

    def begin_transition_preheat(self) -> None:
        """Feed both scene slots into the permanent CUDA transition filter."""
        if self._initial_pgm_scene is None:
            raise RuntimeError("set_initial_scene() must be called before preheating")
        self._transition_prewarm.begin(self._initial_pgm_scene)

    def finish_transition_preheat(self) -> None:
        """Restore direct PGM/PVW routing after transition warm-up."""
        self._transition_prewarm.finish()

    def start_output(self) -> None:
        """Open deferred output for current frames after hidden prewarming."""
        if not self._built:
            raise RuntimeError("build() must precede start_output()")
        if not self._output_started:
            self._transition_prewarm.start_output()
            self._output_started = True

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

            if src.route_router is None:
                self.avp.addNode(OneToMany({
                    "type": "one_to_many",
                    "name": self._n(f"otm_{src.name}"),
                    "src": src.pre_otm_edge,
                    "dst": [self._e(f"{src.name}_a"), self._e(f"{src.name}_b")],
                    "outputs": outputs_init,
                    "timeline": self.timeline,
                    "group": src.input_group,
                }))
                slot_a_edge = self._e(f"{src.name}_a")
                slot_b_edge = self._e(f"{src.name}_b")
            else:
                slot_a_edge = src.pre_filter_edge_a
                slot_b_edge = src.pre_filter_edge_b

            # Default scale: fit to canvas.  MixerOrchestrator rewrites the
            # graph string on every scene switch via node.param.set + auto_restart.
            fallback_graph = (
                f"scale_cuda=w={self.canvas_w}:h={self.canvas_h}:interp_algo=lanczos"
            )
            default_graph = fallback_graph if src.default_graph is None else src.default_graph
            if not default_graph:
                continue

            for slot, edge in (("a", slot_a_edge), ("b", slot_b_edge)):
                self.avp.addNode(FilterVideo({
                    "name": self._n(f"cs_{src.name}_{slot}"),
                    "src": edge,
                    "dst": self._e(f"{src.name}_scaled_{slot}"),
                    "graph": default_graph,
                    "hwaccel": self.hwaccel,
                    "group": src.input_group,
                    "auto_restart": "on",
                }))

    def _source_slot_edge(self, source: MixerSource, slot: str) -> str:
        if source.default_graph != "":
            return self._e(f"{source.name}_scaled_{slot}")
        if source.route_router is not None:
            return source.pre_filter_edge_a if slot == "a" else source.pre_filter_edge_b
        return self._e(f"{source.name}_{slot}")

    def _build_compositors(self) -> None:
        """Build both slots around the native shared output clock."""
        active_pgm = self._active_inputs_mask(self._initial_scene_def())
        timing = {} if self.latency_ms is None else {"latency_ms": self.latency_ms}
        for slot in ("a", "b"):
            is_program = slot.upper() == self._initial_pgm_slot
            self.avp.addNode(CudaRectOverlay({
                "name": self._n(f"comp_{slot}"),
                "src": [self._source_slot_edge(source, slot) for source in self._sources],
                "dst": self._e(f"scene_{slot}_composite"),
                "hwaccel": self.hwaccel,
                "width": self.canvas_w,
                "height": self.canvas_h,
                "sw_format": "nv12",
                "fps": self._fps_str(),
                **timing,
                "scale": any(source.default_graph == "" for source in self._sources),
                "layers": [
                    {k: v for k, v in self._initial_scene_def().sources.get(source.name, {}).items() if k != "graph"}
                    if is_program else {} for source in self._sources
                ],
                "active_inputs": active_pgm if is_program else 0,
                "timeline": self.timeline,
                "group": f"{self.name}_{slot}",
            }))
            self._add_snapshot_node(
                self._n(f"snapshot_{slot}"), self._e(f"scene_{slot}_composite"),
                self._e(f"scene_{slot}_out"), f"{self.name}_{slot}",
                slot=0 if slot == "a" else 1,
            )
            self.avp.addNode(OneToMany({
                "name": self._n(f"otm_scene_{slot}"),
                "src": self._e(f"scene_{slot}_out"),
                "dst": [self._e(f"sc{slot.upper()}_direct"), self._e(f"sc{slot.upper()}_trans")],
                "outputs": 1 if is_program else 0,
                "timeline": self.timeline,
                "group": f"{self.name}_{slot}",
            }))

    def _add_snapshot_node(self, name, source, destination, group, slot=-1):
        timing = {} if self.latency_ms is None else {"latency_ms": self.latency_ms}
        self.avp.addNode(_SnapshotNode({
            "name": name, "src": source, "dst": destination, "group": group,
            "snapshot": self._n("out_sel_snapshot"), "slot": slot,
            "fps": self._fps_str(), **timing,
        }))

    def _build_output_path(self) -> None:
        """Create the output selector and wipe routing nodes."""
        fps_str = self._fps_str()
        pgm_is_a = self._initial_pgm_slot == "A"
        initial_active = 0 if pgm_is_a else 1

        self.avp.addNode(FilterVideo({
            "name": self._n("out_sel_transition"),
            "src": [self._e("scA_trans"), self._e("scB_trans")],
            "dst": self._e("trans_out"),
            "graph": "transition_cuda=alpha='0':eval=frame",
            "hwaccel": self.hwaccel,
            "defer_preliminary_init": True,
            "group": self.name,
        }))

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
            "outputs": 1 if self._output_started else 0,
            "timeline": self.timeline,
            "group": self.name,
        }))
        self.avp.addNode(SourceSwitcher({
            "name": self._n("wipe_sel"),
            "src": [self._e("final_direct"), self._e("wipe_overlay_out")],
            "dst": self._e("final_picture"),
            "active": 0,
            "fallback_active": 0,
            "timeline_reference_input": 0,
            "fallback_when_active_missing": False,
            "timeline": self.timeline,
            "group": self.name,
        }))

        self._add_snapshot_node(self._n("snapshot_output"),
                                self._e("final_picture"), self._e("final_out"), self.name)

    def _build_wipe_subgraph(self) -> None:
        """Create the pre-declared wipe subgraph (not started; orchestrator manages it)."""
        W, H = self.canvas_w, self.canvas_h
        fps_str = self._fps_str()
        wipe_group = f"{self.name}_wipe"
        # With the cache on, everything up to and including the decode lives in a
        # group a take never starts; the take only replays cached frames.
        load_group = clipcache.loader_group(self.name) if self.cache_wipes_mb is not None else wipe_group

        self.avp.addNode(InputRec({
            "name": self._n("wipe_input"),
            "url": "",
            "loop": False,
            "dst": self._e("wipe_raw_pkt"),
            "group": load_group,
        }))
        self.avp.addNode(Demux({
            "name": self._n("wipe_demux"),
            "src": self._e("wipe_raw_pkt"),
            "routing": {"v:0": self._e("wipe_v_pkt")},
            "group": load_group,
        }))
        self.avp.addNode(DecVideo({
            "name": self._n("wipe_dec"),
            "src": self._e("wipe_v_pkt"),
            "dst": self._e("wipe_dec_out"),
            "pixel_format": "?cuda",
            "hwaccel": self.hwaccel,
            "group": load_group,
        }))
        # Alpha media codecs decode on the CPU. Upload the converted wipe to
        # the configured mixer device; hwupload_cuda creates a separate CUDA
        # context that overlay_many_cuda cannot mix with the program frames.
        self.avp.addNode(FilterVideo({
            "name": self._n("wipe_fmt"),
            "src": self._e("wipe_dec_out"),
            "dst": self._e("wipe_fmt_out"),
            # Upload the clip at its own size and let the compositor scale it on
            # the GPU. Resizing to the canvas on a CPU thread cost two thirds of
            # this chain and made the compositor miss 60 Hz ticks during a wipe.
            "graph": "format=rgba,hwupload",
            "hwaccel": self.hwaccel,
            "group": load_group,
        }))
        self.avp.addNode(Realtime({
            "name": self._n("wipe_rt"),
            "src": self._e("wipe_fmt_out"),
            "dst": self._e("wipe_rt_out"),
            "set_pts": True,
            "group": load_group,
        }))
        if self.cache_wipes_mb is not None:
            self.avp.addNode(ClipCache(clipcache.cache_node(
                name=self._n("wipe_cache"), src=self._e("wipe_rt_out"),
                dst=self._e("wipe_cached"), group=wipe_group, fps=fps_str,
                budget_mb=self.cache_wipes_mb)))
        self.avp.addNode(ForceFPS({
            "name": self._n("wipe_rt_fps"),
            "fps": fps_str,
            "src": self._e("wipe_cached") if self.cache_wipes_mb is not None else self._e("wipe_rt_out"),
            "dst": self._e("wipe_rt_fps_out"),
            "group": wipe_group,
        }))
        # The wipe is one alpha-blended layer over the program, drawn by the same
        # compositor kernel the scenes use: no format round trip through
        # yuv420p, no second blend pass and no CPU resize.
        self.avp.addNode(CudaRectOverlay({
            "name": self._n("wipe_overlay"),
            "src": [self._e("final_wipe_in"), self._e("wipe_rt_fps_out")],
            "dst": self._e("wipe_overlay_out"),
            "hwaccel": self.hwaccel,
            "width": W, "height": H, "sw_format": "nv12", "fps": fps_str, "scale": True,
            "layers": [{"dst_x": 0, "dst_y": 0, "dst_w": W, "dst_h": H},
                       {"dst_x": 0, "dst_y": 0, "dst_w": W, "dst_h": H, "z": 1, "blend": True}],
            "active_inputs": 3,
            **({} if self.latency_ms is None else {"latency_ms": self.latency_ms}),
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
            "switch_margin_ms": self.switch_margin_ms,
            "source_switcher": self._n("out_sel"),
            **({"keyframe_node": self.keyframe_node} if self.keyframe_node else {}),
            "initial_pgm_slot": self._initial_pgm_slot,
            "initial_pgm_scene": self._initial_pgm_scene,
            "wipe_otm": self._n("otm_final"),
            "wipe_base_fps": self._n("wipe_base_fps"),
            "wipe_selector": self._n("wipe_sel"),
            "slot_a": {
                "compositor": self._n("comp_a"),
                "post_otm": self._n("otm_scene_a"),
            },
            "slot_b": {
                "compositor": self._n("comp_b"),
                "post_otm": self._n("otm_scene_b"),
            },
        }

        if self.enable_wipe:
            init_cfg.update({
                "wipe_group": wipe_group,
                "wipe_input_node": self._n("wipe_cache" if self.cache_wipes_mb is not None
                                            else "wipe_input"),
                "wipe_tail_edge": self._e("wipe_rt_fps_out"),
                "wipe_flush_edges": wipe_flush_edges,
            })

        lines = [f"mixer.init {self.name} {json.dumps(init_cfg)}"]

        for idx, src in enumerate(self._sources):
            if src.route_router is None:
                lines.append(
                    f"mixer.source {self.name} {src.name}"
                    f" {self._n('otm_' + src.name)} {idx}"
                    + (f" {self._n('cs_' + src.name + '_a')} {self._n('cs_' + src.name + '_b')}"
                       if src.default_graph != "" else "")
                )
            else:
                lines.append(
                    "mixer.routed_source "
                    + json.dumps({
                        "mixer": self.name,
                        "name": src.name,
                        "router": src.route_router,
                        "input_index": idx,
                        "route_label_a": src.route_output_label_a,
                        "route_label_b": src.route_output_label_b,
                        "cs_node_a": self._n("cs_" + src.name + "_a") if src.default_graph != "" else "",
                        "cs_node_b": self._n("cs_" + src.name + "_b") if src.default_graph != "" else "",
                    })
                )

        for scene_name, scene in self._scenes.items():
            lines.append(self._scene_command(scene_name, scene))

        if not self.defer_initial_routes:
            lines.append("mixer.init_routes " + json.dumps({"mixer": self.name}))
            self._routes_initialized = True

        self.avp.executeCommandsFromString("\n".join(lines))

    def _scene_command(self, scene_name: str, scene: MixerScene) -> str:
        scene_def = {"sources": scene.sources}
        if scene.controls:
            scene_def["controls"] = scene.controls
        if scene.routes:
            scene_def["routes"] = scene.routes
        return f"mixer.scene {self.name} {scene_name} {json.dumps(scene_def)}"
