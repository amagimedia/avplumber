"""Scene definitions and source routing descriptions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MixerSource:
    name: str
    pre_otm_edge: Optional[str]
    input_group: str
    default_graph: Optional[str] = None
    pre_filter_edge_a: Optional[str] = None
    pre_filter_edge_b: Optional[str] = None
    route_router: Optional[str] = None
    route_output_label_a: Optional[str] = None
    route_output_label_b: Optional[str] = None


@dataclass
class MixerScene:
    name: str
    # source_name -> {"graph": ..., "dst_x": ..., "dst_y": ..., ...}
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    controls: List[Dict[str, Any]] = field(default_factory=list)
    routes: Dict[str, int] = field(default_factory=dict)


