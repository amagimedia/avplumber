"""Scene definitions and source routing descriptions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MixerSource:
    name: str
    pre_otm_edge: Optional[str]
    input_group: str
    pre_filter_edge_a: Optional[str] = None
    pre_filter_edge_b: Optional[str] = None
    route_router: Optional[str] = None
    route_output_label_a: Optional[str] = None
    route_output_label_b: Optional[str] = None
    color: Any = None
    pixel_format: Optional[str] = None
    packed_rgb: bool = False
    premultiplied_alpha: bool = False
    color_tagged: bool = False


@dataclass
class MixerScene:
    name: str
    # source_name -> compositor geometry
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    controls: List[Dict[str, Any]] = field(default_factory=list)
    routes: Dict[str, int] = field(default_factory=dict)


