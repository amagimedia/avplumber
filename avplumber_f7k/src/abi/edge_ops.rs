//! Edge ops C ABI. take + peek guard over owned Media.

use crate::abi::convert::media_to_avp;
use crate::abi::{AvpBuffer, AvpSpec};
use crate::abi::{AvpEdge, AvpNode};
use crate::graph::edge::{EdgeEvent, EdgeItem, Push};
use crate::graph::spec::Spec;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum AvpEventType {
    Eof = 1,
    FlushStart = 2,
    FlushStop = 3,
    Spec = 4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpEdgeEvent {
    pub r#type: AvpEventType,
    pub spec: AvpSpec,
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
        EdgeEvent::Eof => AvpEdgeEvent {
            r#type: AvpEventType::Eof,
            spec: AvpSpec::zeroed(),
        },
        EdgeEvent::FlushStart => AvpEdgeEvent {
            r#type: AvpEventType::FlushStart,
            spec: AvpSpec::zeroed(),
        },
        EdgeEvent::FlushStop => AvpEdgeEvent {
            r#type: AvpEventType::FlushStop,
            spec: AvpSpec::zeroed(),
        },
        EdgeEvent::Spec(s) => AvpEdgeEvent {
            r#type: AvpEventType::Spec,
            spec: AvpSpec::from(s),
        },
    }
}

fn item_to_c(item: EdgeItem) -> AvpItem {
    match item {
        EdgeItem::Buffer(m) => AvpItem {
            is_event: 0,
            buffer: media_to_avp(m, &Default::default()),
            event: AvpEdgeEvent {
                r#type: AvpEventType::Eof,
                spec: AvpSpec::zeroed(),
            },
        },
        EdgeItem::Event(e) => AvpItem {
            is_event: 1,
            buffer: AvpBuffer::null(crate::graph::AvpMediaType::VIDEO),
            event: event_to_c(&e),
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
    let edge = unsafe { &*edge };
    let buf = unsafe { *buf };
    // Without Instance we cannot reconstruct Opaque vtables; ffmpeg/stub adopt the pointer.
    let Some(media) = avp_to_media_loose(buf) else {
        return AvpFlow::Error;
    };
    let result = match crate::abi::ffi_node::callback_generation() {
        Some(generation) => match edge.edge.offer_generation(generation, media) {
            Ok(()) => Push::Accepted,
            Err((status, _)) => status,
        },
        None => edge.edge.push(media),
    };
    push_to_c(result)
}

fn avp_to_media_loose(buf: AvpBuffer) -> Option<crate::graph::Media> {
    #[cfg(feature = "ffmpeg")]
    {
        use crate::graph::Media;
        if buf.is_null() {
            return None;
        }
        match buf.media {
            crate::graph::AvpMediaType::PACKET => {
                let p = std::ptr::NonNull::new(buf.ptr as *mut rusty_ffmpeg::ffi::AVPacket)?;
                Some(Media::Packet(unsafe {
                    rsmpeg::avcodec::AVPacket::from_raw(p)
                }))
            }
            crate::graph::AvpMediaType::VIDEO => {
                let p = std::ptr::NonNull::new(buf.ptr as *mut rusty_ffmpeg::ffi::AVFrame)?;
                Some(Media::Video(unsafe {
                    rsmpeg::avutil::AVFrame::from_raw(p)
                }))
            }
            crate::graph::AvpMediaType::AUDIO => {
                let p = std::ptr::NonNull::new(buf.ptr as *mut rusty_ffmpeg::ffi::AVFrame)?;
                Some(Media::Audio(unsafe {
                    rsmpeg::avutil::AVFrame::from_raw(p)
                }))
            }
            _ => None,
        }
    }
    #[cfg(not(feature = "ffmpeg"))]
    {
        Some(crate::graph::Media::Stub {
            kind: buf.media,
            pts: buf.ptr as usize as i64,
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_edge_push_event(edge: *mut AvpEdge, ev: *const AvpEdgeEvent) {
    let edge = unsafe { &*edge };
    let ev_c = unsafe { *ev };
    let ev = match ev_c.r#type {
        AvpEventType::Eof => EdgeEvent::Eof,
        AvpEventType::FlushStart => EdgeEvent::FlushStart,
        AvpEventType::FlushStop => EdgeEvent::FlushStop,
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
                *out = item_to_c(item);
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
                *out = item_to_c(item.clone());
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
    if !out.is_null() {
        if let Some(EdgeItem::Buffer(m)) = p.cloned {
            unsafe {
                *out = media_to_avp(m, &Default::default());
            }
            return 1;
        }
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
