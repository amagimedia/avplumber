//! Numeric IDs for node capabilities and core services.
//!
//! This file is the source of truth. `include/avplumber_ids.h` is generated
//! from it (cbindgen). Callers query a node they already hold
//! (`avp_node_query_interface`) or a core service (`avp_core_query_service`).
//! Per-interface / per-service vtables stay handwritten in
//! `avplumber_interfaces.h` / `avplumber_services_*.h` — those structs have
//! no Rust twin.
//!
//! Stream format is not a capability: it is `Spec` on the edge. Seek/speed/
//! pause are not capabilities: they are core services.

/// Live methods on a specific node (decoder discard, sentinel stats,
/// `node.param` get/set, demux stream list). Query the node you already hold;
/// there is no upstream walk.
///
/// Not in this enum: stream format (`Spec` on the edge) or playback control
/// (`avp_core_query_service`).
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AvpInterfaceId {
    Decoder = 1,
    Sentinel = 2,
    ReturnsObjects = 3,
    InputsObjects = 4,
    StreamsInput = 5,
}

impl AvpInterfaceId {
    pub fn from_raw(id: u32) -> Option<Self> {
        let all = [
            Self::Decoder,
            Self::Sentinel,
            Self::ReturnsObjects,
            Self::InputsObjects,
            Self::StreamsInput,
        ];
        all.into_iter().find(|iface| *iface as u32 == id)
    }
}

/// Core-owned services. An embedder looks them up with
/// `avp_core_query_service`; each service's ID and vtable live in
/// `avplumber_services_*.h`.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AvpServiceId {
    Clock = 1,
    Timeline = 2,
}

impl AvpServiceId {
    pub fn from_raw(id: u32) -> Option<Self> {
        let all = [Self::Clock, Self::Timeline];
        all.into_iter().find(|svc| *svc as u32 == id)
    }
}
