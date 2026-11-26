"""The mixer's REST routes: `/mixer` for `pyplumber.rest.create_app`.

Each route is one of the mixer's control commands (`mixer.*`) with a typed request. They go
through a `MixerBridge`, so a gateway that shares the control page's bridge also shares its
ordering of takes and its cached state. Requests are strict: a field the model does not name is
refused, never passed on.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Callable, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyplumber.composition.motion import FADE_CURVES, MOVE_UNMATCHED
from pyplumber.rest.client import ControlClient
from pyplumber.rest.errors import ApiError, Starting, refused_while

from .audio_gain import DB_MAX, DB_MIN
from .audio_processors import DUCKER_RANGES
from .config import MAX_KEY_FADE_SECONDS
from .gui.bridge import MixerBridge

log = logging.getLogger(__name__)

Curve = Literal[FADE_CURVES]   # a tuple subscript: Literal["linear", "ease-in", ...]
Unmatched = Literal[MOVE_UNMATCHED]
Scene = Annotated[str, Field(min_length=1, description="A scene id, one of `GET /mixer/scenes`")]
Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$", description="Opaque RGB, `#RRGGBB`")]
Seconds = Annotated[float, Field(gt=0)]
KeyFadeSeconds = Annotated[float, Field(ge=0, le=MAX_KEY_FADE_SECONDS, description="0 cuts")]
ShowTick = Annotated[int, Field(description="Reach Program no earlier than this show tick (two hosts, one frame); "
                                            "the engine needs AVP_CLOCK_ORIGIN=tai")]


class ClientBridge(MixerBridge):
    """A bridge whose commands go to `client` instead of a control connection of its own: the fake
    engine (`pyplumber.rest.fake`) and tests. Ordering of takes and state caching are MixerBridge's."""

    def __init__(self, client: ControlClient, mixer: str = "mixer", timeout: float = 10.0):
        # MixerBridge.__init__ opens connections and starts their event loop; none is needed here.
        self.mixer, self.timeout, self.transition, self._client = mixer, timeout, None, client
        self._init_sharing()

    def command(self, line: str, timeout: float | None = None) -> str | None:
        return self._client.command(line, timeout)


class Request(BaseModel):
    # strict: "1" is not a number and true is not 1, as the mixer's own checks have it.
    model_config = ConfigDict(extra="forbid", strict=True)


class Reply(BaseModel):
    """The fields a client can rely on; the mixer reports more (timing, telemetry), passed through."""
    model_config = ConfigDict(extra="allow")


# --- Program and Preview ---

class Lever(Reply):
    position: float
    engaged: bool


class Status(Reply):
    pgm_scene: str
    pvw_scene: str = Field(description="Shown to the operator; after a take, the scene that left Program")
    pvw_slot_scene: str = Field("", description="Loaded in the Preview slot, so a take of it is instant; empty while cold")
    transition: Literal["idle", "cut", "crossfade", "wipe", "move"]
    transition_on_air: bool | None = None
    transition_progress: float | None = None
    transition_paused: bool | None = None
    transition_manual: bool | None = Field(None, description="The lever runs the transition")
    transition_duration_ms: int | None = None
    lever: Lever | None = None


class Scenes(BaseModel):
    scenes: list[str]


class Preview(Request):
    scene: Scene


class Zoom(Request):
    """Glide one person's layer from Program's framing to the scene's instead of cutting, when both
    scenes draw one layer of the same source; otherwise the cut is plain and the status `zoom` says why."""
    duration_sec: float = Field(ge=0.1, le=10)
    curve: Curve = "ease-in-out"


class Cut(Request):
    type: Literal["cut"]
    scene: Scene
    zoom: Zoom | None = None
    at_show_tick: ShowTick | None = None


class Fade(Request):
    type: Literal["fade"]
    scene: Scene
    duration_sec: Seconds | None = Field(None, description="Default 1")
    curve: Curve | None = Field(None, description="Default linear")
    color: Color | None = Field(None, description="Dip through this colour instead of mixing")
    at_show_tick: ShowTick | None = None


class Wipe(Request):
    type: Literal["wipe"]
    scene: Scene
    wipe_file: str = Field(min_length=1, description="The clip's path on the engine's host; the show's are in the settings' `wipes`")
    duration_sec: Seconds | None = Field(None, description="Default: the clip's length")


class Move(Request):
    """Layers of a source in both scenes glide from Program's placement to the scene's; it fades
    when it cannot glide, and the status `move` says why."""
    type: Literal["move"]
    scene: Scene
    duration_sec: Seconds | None = Field(None, description="Default 1")
    curve: Curve | None = Field(None, description="Default linear")
    unmatched: Unmatched | None = Field(None, description="What sources in only one scene do; default fade")
    at_show_tick: ShowTick | None = None


Take = Annotated[Cut | Fade | Wipe | Move, Field(discriminator="type")]


class TakeResult(BaseModel):
    superseded: bool = Field(description="A newer take arrived before this one reached the mixer, which took that one instead")


class LeverPosition(Request):
    """The lever arm (T-bar). From the end it rests at it mixes to Preview, the other end completes,
    its start cancels. Positions sent faster than the mixer takes them collapse to the newest."""
    position: float = Field(ge=0, le=1)
    duration_sec: Seconds | None = Field(None, description="The mix's length, for AUTO to run it on from the lever")
    curve: Curve | None = None
    color: Color | None = None
    wipe_file: str | None = Field(None, description="Wipe with this cached clip instead of mixing, its frame following the lever")
    move: bool | None = Field(None, description="The lever's take is a move")
    unmatched: Unmatched | None = None
    t_ms: float | None = Field(None, description="When the controller sampled the position, on its own clock: paces positions against network jitter")
    seq: int | None = Field(None, description="Numbered per `client`; a position older than one already received is dropped")
    client: str | None = None


# --- Keys (DSK) and fade to black ---

class Key(Reply):
    id: str
    source: str
    on: bool
    rate: float | None = Field(description="The key's own fade length, null while it follows `all_rate`")
    all_rate: float


class Keys(BaseModel):
    keys: list[Key]


class KeyChange(Request):
    """Either put the key on or off air (`on`, over `fade_seconds`), or set its own rate (`rate`;
    null: it follows the rate of all keys again). A rate changes nothing on air."""
    on: bool | None = None
    fade_seconds: KeyFadeSeconds | None = Field(None, description="Default: the key's rate, or else the rate of all keys")
    curve: Curve | None = None
    rate: KeyFadeSeconds | None = None

    @model_validator(mode="after")
    def _one_change(self):
        rate = "rate" in self.model_fields_set
        if rate == (self.on is not None):
            raise ValueError("give either on or rate")
        if rate and (self.fade_seconds is not None or self.curve is not None):
            raise ValueError("fade_seconds and curve belong to on")
        return self


class KeyGroupChange(Request):
    """Several keys on one program frame: `on`, `toggle` (each against what is on air), or `prepare`:
    the keys that switch with the following background takes, on the take's clock (an empty list
    clears the selection; `fade_seconds` and `curve` do not apply to it)."""
    keys: list[str]
    on: bool | None = None
    toggle: bool = False
    prepare: bool = False
    fade_seconds: KeyFadeSeconds | None = Field(None, description="Default: the rate of all keys")
    curve: Curve | None = None

    @model_validator(mode="after")
    def _one_change(self):
        if (self.on is not None) + self.toggle + self.prepare != 1:
            raise ValueError("give exactly one of on, toggle and prepare")
        if not self.prepare and not self.keys:
            raise ValueError("keys must name at least one key")
        return self


class KeyRate(Request):
    seconds: KeyFadeSeconds


class Ftb(Reply):
    on: bool


class FtbChange(Request):
    """Fade the keyed Program (not the clean feed) to black, or back."""
    on: bool | None = None
    toggle: bool = False
    fade_seconds: KeyFadeSeconds | None = Field(None, description="Default: the show's fade length")
    curve: Curve | None = None

    @model_validator(mode="after")
    def _one_change(self):
        if (self.on is not None) + self.toggle != 1:
            raise ValueError("give either on or toggle")
        return self


# --- AUX buses ---

class AuxBus(Reply):
    id: str
    revision: str = Field(description="Changes with every assignment or layout change; `PUT …/scenes` names the one it edits")
    scenes: list[str | None]
    layout: dict[str, Any]


class AuxBuses(BaseModel):
    buses: list[AuxBus]


class AuxChanged(Reply):
    revision: str
    scenes: list[str | None]
    layout: dict[str, Any]
    layouts: list[dict[str, Any]] = Field(description="The layouts the bus offers")


class AuxScenes(Request):
    scenes: list[Scene | None] = Field(description="Every slot's scene, by slot index, as the bus reports them; null clears one")
    expected_revision: str


class AuxLayout(Request):
    layout: dict[str, Any] = Field(description='One of the bus\'s `layouts`: `{"preset": …}` or `{"cells": […]}`')


class AuxPage(Request):
    page: int | None = Field(None, ge=0)
    step: int | None = Field(None, description="Relative, wrapping around")

    @model_validator(mode="after")
    def _one_change(self):
        if (self.page is None) == (self.step is None):
            raise ValueError("give either page or step")
        return self


# --- Sources ---

class SourceSet(Request):
    """Repoint a wired file / live video (or NDI) source: stop its group safely, set URL and loop,
    restart. Browser, raw, MXL and VMM pads refuse the command."""
    url: Annotated[str, Field(min_length=1)]
    loop: bool


class SourceSetResult(Reply):
    id: str
    url: str
    loop: bool


class SourcePrepare(Request):
    """Load a URL on a playback-enabled pad, pause on the cue-in frame. Needs
    `MixerOptions.playback_control`."""
    url: Annotated[str, Field(min_length=1)]
    loop: bool
    cue_in_ms: int = Field(0, ge=0, description="Seek to this offset after pause, in milliseconds")


class SourcePrepareResult(Reply):
    id: str
    url: str
    loop: bool
    cue_in_ms: int
    duration_ms: int | None = Field(None, description="ffprobe length when known; null for live / unprobeable")


class SourceMedia(Request):
    """play / pause / seek / stop / restart on a playback-enabled source."""
    action: Literal["play", "pause", "seek", "stop", "restart"]
    position_ms: int | None = Field(None, ge=0, description="Required for seek; ignored otherwise")

    @model_validator(mode="after")
    def _seek_needs_position(self):
        if self.action == "seek" and self.position_ms is None:
            raise ValueError("seek needs position_ms")
        if self.action != "seek" and self.position_ms is not None:
            raise ValueError("position_ms is only valid with action seek")
        return self


class SourceMediaResult(Reply):
    id: str
    action: str
    position_ms: int | None = None


class SourceResumeAt(Request):
    """Schedule resume of the source's pause team at a wallclock time (ms)."""
    at_ms: int = Field(ge=0, description="Wallclock ms in the mixer start_pts domain")
    position_ms: int | None = Field(
        None, ge=0,
        description="Client playhead at resume; echoed only — the engine does not track position")


class SourceResumeAtResult(Reply):
    id: str
    at_ms: int
    position_ms: int | None = None


# --- Audio ---

GainDb = Annotated[float, Field(ge=DB_MIN, le=DB_MAX)]


class Gain(BaseModel):
    id: str
    gain_db: float


class Gains(BaseModel):
    gains: list[Gain]


class GainsChange(Request):
    gains: dict[str, GainDb] = Field(min_length=1, description="Source id to its volume in dB, independent of audio-follow-video")


class Processor(Reply):
    output: str
    id: str
    enabled: bool
    gain_db: float = Field(description="The gain reduction applied now")


class Processors(BaseModel):
    processors: list[Processor]


def _ranged(name: str):
    low, high = DUCKER_RANGES[name]
    return Annotated[float | None, Field(ge=low, le=high)]


class ProcessorChange(Request):
    enabled: bool | None = None
    threshold_db: _ranged("threshold_db") = None
    depth_db: _ranged("depth_db") = None
    attack_ms: _ranged("attack_ms") = None
    hold_ms: _ranged("hold_ms") = None
    release_ms: _ranged("release_ms") = None

    @model_validator(mode="after")
    def _something(self):
        if not self.model_dump(exclude_defaults=True):
            raise ValueError("nothing to set")
        return self


def mixer_router(bridge: MixerBridge, *, on_aux_change: Callable[[str, dict], None] | None = None,
                 starting: Starting | None = None) -> APIRouter:
    """`/mixer` for the mixer `bridge` controls.

    `on_aux_change(bus, result)` is called after an accepted AUX change, for an application that
    persists them (`SetupController.remember_aux`). An exception from it is logged: the change is
    on the bus already, and the request answers with it.

    `starting`: as `create_app`'s, for an application that restarts the mixer: the routes answer
    503 `engine_not_ready` while it gives a reason.
    """
    router = APIRouter(prefix="/mixer", tags=["mixer"], dependencies=refused_while(starting))

    def send(command: str, request: Request | None = None, **fields):
        """`bridge.take` is looked up on each call: an application may have wrapped it."""
        body = request.model_dump(exclude_defaults=True, exclude={"type"}) if request else {}
        try:
            return bridge.take({"command": command, **body, **fields})
        except json.JSONDecodeError:
            raise
        except ValueError as exc:   # the mixer answered {"error": …}: an edit against a stale revision
            raise ApiError(409, "conflict", str(exc)) from None

    def ask(line: str, empty: str):
        return json.loads(bridge.command(line) or empty)

    # --- Program and Preview ---

    @router.get("/status", response_model=Status)
    def status():
        """What is on Program and Preview, and the transition on air."""
        return bridge.status()

    @router.get("/scenes", response_model=Scenes)
    def scenes():
        return {"scenes": ask(f"mixer.scenes {bridge.mixer}", "[]")}

    @router.get("/settings", response_model=dict[str, Any])
    def settings():
        """What the show fixed when it was built: canvas, sources, keys, AUX buses, wipes, outputs."""
        return bridge.settings()

    @router.put("/preview", status_code=204)
    def preview(request: Preview):
        send("preview", request)

    @router.post("/takes", response_model=TakeResult)
    def take(request: Take):
        """Take a scene to Program. Takes reach the mixer one at a time, in arrival order, and a
        newer one supersedes the one still waiting. While a transition is on air, a take acts on
        it as a vision switcher's buttons do: a cut completes it."""
        return {"superseded": send(request.type, request) is False}

    @router.post("/transition/interrupt", status_code=204)
    def interrupt():
        """Drop an armed or running transition and keep the picture on Program."""
        send("interrupt")

    @router.put("/lever", status_code=204)
    def lever(request: LeverPosition):
        send("lever", request)

    # --- Keys (DSK) and fade to black ---

    @router.get("/keys", response_model=Keys)
    def keys():
        """The downstream keys as the last program frame rendered them."""
        return {"keys": bridge.keys()["dsk"]}

    @router.patch("/keys", response_model=Keys)
    def change_keys(request: KeyGroupChange):
        return {"keys": send("dsk", request)}

    @router.put("/keys/rate", response_model=Keys)
    def set_keys_rate(request: KeyRate):
        """The fade length of every key without a rate of its own."""
        return {"keys": send("dsk", all_rate=request.seconds)}

    @router.patch("/keys/{key}", response_model=Keys)
    def change_key(key: str, request: KeyChange):
        if "rate" in request.model_fields_set:
            return {"keys": send("dsk", key=key, rate=request.rate)}   # null is a value here
        return {"keys": send("dsk", request, key=key)}

    @router.get("/ftb", response_model=Ftb)
    def ftb():
        return bridge.keys()["ftb"]

    @router.put("/ftb", response_model=Ftb)
    def change_ftb(request: FtbChange):
        return send("ftb", request)

    # --- AUX buses ---

    def aux(command: str, bus: str, request: Request):
        result = send(command, request, bus=bus)
        if on_aux_change:
            try:
                on_aux_change(bus, result)
            except Exception:   # the bus is live; only what the application does with it failed
                log.exception("on_aux_change failed for aux %s", bus)
        return result

    @router.get("/aux", response_model=AuxBuses)
    def aux_buses():
        return {"buses": ask("mixer.aux_status", "[]")}

    @router.put("/aux/{bus}/scenes", response_model=AuxChanged)
    def assign_aux_scenes(bus: str, request: AuxScenes):
        """Answers 409 `conflict` when the bus changed since `expected_revision`: read it again."""
        return aux("aux", bus, request)

    @router.put("/aux/{bus}/layout", response_model=AuxChanged)
    def set_aux_layout(bus: str, request: AuxLayout):
        return aux("aux_layout", bus, request)

    @router.put("/aux/{bus}/page", response_model=AuxChanged)
    def turn_aux_page(bus: str, request: AuxPage):
        """Show a page of a bus whose layout pages through the sources."""
        return aux("aux_page", bus, request)

    # --- Sources ---

    @router.put("/sources/{source_id}", response_model=SourceSetResult)
    def set_source(source_id: str, request: SourceSet):
        """Change the URL and loop of an existing show source pad without EOS killing the mixer."""
        return send("source.set", request, id=source_id, mixer=bridge.mixer)

    @router.post("/sources/{source_id}/prepare", response_model=SourcePrepareResult)
    def prepare_source(source_id: str, request: SourcePrepare):
        """Load a URL, pause on the cue-in frame. Needs a mixer built with playback_control."""
        return send("source.prepare", request, id=source_id, mixer=bridge.mixer)

    @router.post("/sources/{source_id}/media", response_model=SourceMediaResult,
                 response_model_exclude_none=True)
    def source_media(source_id: str, request: SourceMedia):
        """play / pause / seek / stop / restart. Playhead stays client-estimated."""
        return send("source.media", request, id=source_id, mixer=bridge.mixer)

    @router.post("/sources/{source_id}/resume_at", response_model=SourceResumeAtResult,
                 response_model_exclude_none=True)
    def resume_source_at(source_id: str, request: SourceResumeAt):
        """Schedule `resume <pause_team> at <at_ms>` for a timed cue-in."""
        return send("source.resume_at", request, id=source_id, mixer=bridge.mixer)

    # --- Audio ---

    @router.get("/audio/gains", response_model=Gains)
    def audio_gains():
        return {"gains": bridge.audio_gain()}

    @router.patch("/audio/gains", response_model=Gains)
    def set_audio_gains(request: GainsChange):
        return {"gains": send("audio_gain", request)}

    @router.get("/audio/processors", response_model=Processors)
    def audio_processors():
        return {"processors": bridge.audio_processors()}

    @router.patch("/audio/processors/{output}/{processor}", response_model=Processors)
    def set_audio_processor(output: str, processor: str, request: ProcessorChange):
        return {"processors": send("audio_processor", request, output=output, id=processor)}

    return router
