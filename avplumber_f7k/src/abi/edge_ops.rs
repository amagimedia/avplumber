//! Edge ops C ABI. take + peek guard over owned Media.

use crate::abi::convert::{clone_avp_buffer, media_as_avp, media_to_avp, release_avp_buffer};
use crate::abi::{AvpBuffer, AvpSpec};
use crate::abi::{AvpEdge, AvpNode};
use crate::graph::edge::{EdgeEvent, EdgeItem, Push};
use crate::graph::media::Ts;
use crate::graph::spec::Spec;
use crate::graph::{AVP_NOPTS, AvpRational};

#[repr(C)]
#[derive(Clone, Copy)]
pub enum AvpEventType {
    Eof = 1,
    FlushStart = 2,
    FlushStop = 3,
    Spec = 4,
    /// Emit what a codec is holding, without ending the stream.
    Drain = 5,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpEdgeEvent {
    pub r#type: AvpEventType,
    pub spec: AvpSpec,
    /// `FlushStop`: the media time the source aimed for, `AVP_NOPTS` when the
    /// reposition was exact. Its time base is `resume_at_tb`.
    pub resume_at: i64,
    pub resume_at_tb: AvpRational,
}

impl AvpEdgeEvent {
    /// An event with no payload of either kind.
    fn plain(r#type: AvpEventType) -> Self {
        Self {
            r#type,
            spec: AvpSpec::zeroed(),
            resume_at: AVP_NOPTS,
            resume_at_tb: AvpRational::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpItem {
    pub is_event: i32,
    pub buffer: AvpBuffer,
    pub event: AvpEdgeEvent,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AvpFlow {
    Pushed = 0,
    Drop = 1,
    Backpressure = 2,
    Eof = 3,
    Error = 4,
}

fn push_to_c(p: Push) -> AvpFlow {
    match p {
        Push::Accepted => AvpFlow::Pushed,
        Push::Dropped => AvpFlow::Drop,
        Push::Full => AvpFlow::Backpressure,
        Push::Closed => AvpFlow::Eof,
    }
}

fn event_to_c(ev: &EdgeEvent) -> AvpEdgeEvent {
    match ev {
        EdgeEvent::Eof => AvpEdgeEvent::plain(AvpEventType::Eof),
        EdgeEvent::FlushStart => AvpEdgeEvent::plain(AvpEventType::FlushStart),
        EdgeEvent::Drain => AvpEdgeEvent::plain(AvpEventType::Drain),
        EdgeEvent::FlushStop { resume_at } => {
            let mut event = AvpEdgeEvent::plain(AvpEventType::FlushStop);
            if let Some(ts) = resume_at {
                event.resume_at = ts.val;
                event.resume_at_tb = ts.tb;
            }
            event
        }
        EdgeEvent::Spec(s) => AvpEdgeEvent {
            spec: AvpSpec::from(s),
            ..AvpEdgeEvent::plain(AvpEventType::Spec)
        },
    }
}

fn owned_item_to_c(item: EdgeItem) -> AvpItem {
    match item {
        EdgeItem::Buffer(m) => AvpItem {
            is_event: 0,
            buffer: media_to_avp(m),
            event: AvpEdgeEvent::plain(AvpEventType::Eof),
        },
        EdgeItem::Event(e) => AvpItem {
            is_event: 1,
            buffer: AvpBuffer::null(crate::graph::AvpMediaType::VIDEO),
            event: event_to_c(&e),
        },
    }
}

fn borrowed_item_to_c(item: &EdgeItem) -> AvpItem {
    match item {
        EdgeItem::Buffer(media) => AvpItem {
            is_event: 0,
            buffer: media_as_avp(media),
            event: AvpEdgeEvent::plain(AvpEventType::Eof),
        },
        EdgeItem::Event(event) => AvpItem {
            is_event: 1,
            buffer: AvpBuffer::null(crate::graph::AvpMediaType::VIDEO),
            event: event_to_c(event),
        },
    }
}

pub struct AvpPeek {
    pub edge: std::sync::Arc<dyn crate::graph::Edge>,
    pub cloned: Option<EdgeItem>,
    pub generation: Option<u64>,
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_push(edge: *mut AvpEdge, buf: *const AvpBuffer) -> AvpFlow {
    if edge.is_null() || buf.is_null() {
        return AvpFlow::Error;
    }
    let edge = unsafe { &*edge };
    let buf = unsafe { *buf };
    let opaque_vtable = edge.media_vtables.lock().unwrap().get(&buf.media).copied();
    let Some(media) = clone_avp_buffer(buf, opaque_vtable) else {
        return AvpFlow::Error;
    };
    let result = match crate::abi::ffi_node::callback_generation() {
        Some(generation) => match edge.edge.offer_generation(generation, media) {
            Ok(()) => Push::Accepted,
            Err((status, _)) => status,
        },
        None => edge.edge.push(media),
    };
    if result == Push::Accepted {
        let released = release_avp_buffer(buf, opaque_vtable);
        debug_assert!(released);
    }
    push_to_c(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_push_event(edge: *mut AvpEdge, ev: *const AvpEdgeEvent) {
    let edge = unsafe { &*edge };
    let ev_c = unsafe { *ev };
    let ev = match ev_c.r#type {
        AvpEventType::Eof => EdgeEvent::Eof,
        AvpEventType::Drain => EdgeEvent::Drain,
        AvpEventType::FlushStart => EdgeEvent::FlushStart,
        AvpEventType::FlushStop => EdgeEvent::FlushStop {
            resume_at: (ev_c.resume_at != AVP_NOPTS).then(|| Ts {
                val: ev_c.resume_at,
                tb: ev_c.resume_at_tb,
            }),
        },
        AvpEventType::Spec => EdgeEvent::Spec(ev_c.spec.to_native()),
    };
    if let Some(generation) = crate::abi::ffi_node::callback_generation() {
        edge.edge.push_event_generation(generation, ev);
    } else {
        edge.edge.push_event(ev);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_take(edge: *mut AvpEdge, timeout_ms: i32, out: *mut AvpItem) -> i32 {
    let edge = unsafe { &*edge };
    let generation = crate::abi::ffi_node::callback_generation();
    let item = match generation {
        Some(generation) => edge.edge.take_generation(generation, timeout_ms),
        None => edge.edge.take(timeout_ms),
    };
    match item {
        Some(item) => {
            unsafe {
                *out = owned_item_to_c(item);
            }
            1
        }
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_peek(
    edge: *mut AvpEdge,
    timeout_ms: i32,
    out: *mut AvpItem,
) -> *mut AvpPeek {
    let edge = unsafe { &*edge };
    let generation = crate::abi::ffi_node::callback_generation();
    let item = match generation {
        Some(generation) => edge.edge.peek_clone_generation(generation, timeout_ms),
        None => edge.edge.peek_clone(timeout_ms),
    };
    match item {
        Some(item) => {
            unsafe {
                *out = borrowed_item_to_c(&item);
            }
            Box::into_raw(Box::new(AvpPeek {
                edge: edge.edge.clone(),
                cloned: Some(item),
                generation,
            }))
        }
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_peek_release(peek: *mut AvpPeek) {
    if peek.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(peek));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_peek_consume(peek: *mut AvpPeek, out: *mut AvpBuffer) -> i32 {
    if peek.is_null() {
        return 0;
    }
    let p = unsafe { Box::from_raw(peek) };
    if let Some(generation) = p.generation {
        p.edge.pop_generation(generation);
    } else {
        p.edge.pop();
    }
    if !out.is_null()
        && let Some(EdgeItem::Buffer(m)) = p.cloned
    {
        unsafe {
            *out = media_to_avp(m);
        }
        return 1;
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_pop(edge: *mut AvpEdge) {
    let edge = unsafe { &*edge };
    if let Some(generation) = crate::abi::ffi_node::callback_generation() {
        edge.edge.pop_generation(generation);
    } else {
        edge.edge.pop();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_occupied(edge: *mut AvpEdge) -> i32 {
    unsafe { (*edge).edge.occupied() as i32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_current_spec(edge: *mut AvpEdge, out: *mut AvpSpec) -> i32 {
    match unsafe { (*edge).edge.current_spec() } {
        Some(s) => {
            unsafe {
                *out = AvpSpec::from(&s);
            }
            1
        }
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_notify_readable(edge: *mut AvpEdge, _node: *mut AvpNode) {
    unsafe {
        (*edge).edge.arm_readable();
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_notify_writable(edge: *mut AvpEdge, _node: *mut AvpNode) {
    unsafe {
        (*edge).edge.arm_writable();
    }
}

#[allow(dead_code)]
fn _spec_used(_: Spec) {}
