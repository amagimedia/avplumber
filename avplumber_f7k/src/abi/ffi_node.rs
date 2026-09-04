//! C node vtable wrapped as a native `Node`.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::capability::AvpInterfaceId;
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::node::{Blocked, Node, NodeBody, NodeKind, Tick};
use crate::graph::poll_ctx::NodePollContext;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpNodeVtable {
    pub start: Option<extern "C" fn(*mut c_void)>,
    pub stop: Option<extern "C" fn(*mut c_void)>,
    pub destroy: Option<extern "C" fn(*mut c_void)>,
    pub process: Option<extern "C" fn(*mut c_void) -> i32>,
    pub poll: Option<extern "C" fn(*mut c_void) -> i32>,
    pub query_interface: Option<extern "C" fn(*mut c_void, u32) -> *const c_void>,
}

pub struct FfiNode {
    name: String,
    handle: *mut c_void,
    self_ptr: *mut c_void,
    vtable: &'static AvpNodeVtable,
    generation: AtomicU64,
}

unsafe impl Send for FfiNode {}
unsafe impl Sync for FfiNode {}

impl FfiNode {
    pub fn new(
        name: String,
        handle: *mut c_void,
        self_ptr: *mut c_void,
        vtable: &'static AvpNodeVtable,
    ) -> Self {
        Self {
            name,
            handle,
            self_ptr,
            vtable,
            generation: AtomicU64::new(0),
        }
    }

    fn call<R>(&self, f: impl FnOnce() -> R) -> R {
        with_callback_impl(
            self.handle,
            self.self_ptr,
            self.generation.load(Ordering::Acquire),
            f,
        )
    }

    fn flow_blocked(&self, code: i32) -> Result<Blocked, NodeError> {
        match code {
            0..=2 => Ok(Blocked::Again),
            3 => Ok(Blocked::Done),
            4 => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                "C node returned AVP_FLOW_ERROR",
            )),
            _ => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                format!("C node returned invalid AvpFlow code {code}"),
            )),
        }
    }
    fn flow_tick(&self, code: i32) -> Result<Tick, NodeError> {
        match code {
            0 | 1 => Ok(Tick::Again),
            2 => Ok(Tick::Idle),
            3 => Ok(Tick::Done),
            4 => Err(NodeError::new(
                &self.name,
                NodePhase::Poll,
                "C node returned AVP_FLOW_ERROR",
            )),
            _ => Err(NodeError::new(
                &self.name,
                NodePhase::Poll,
                format!("C node returned invalid AvpFlow code {code}"),
            )),
        }
    }

    fn process_result(&self) -> Result<Blocked, NodeError> {
        match self.vtable.process {
            Some(f) => self.flow_blocked(self.call(|| f(self.handle))),
            None => Ok(Blocked::Done),
        }
    }

    fn poll_result(&self) -> Result<Tick, NodeError> {
        match self.vtable.poll {
            Some(f) => self.flow_tick(self.call(|| f(self.handle))),
            None => Ok(Tick::Done),
        }
    }
}

thread_local! {
    static CALLBACK_IMPL: Cell<Option<(usize, usize, u64)>> = const { Cell::new(None) };
    static FACTORY_HANDLE: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(crate) fn with_factory_handle<R>(handle: *mut c_void, f: impl FnOnce() -> R) -> R {
    FACTORY_HANDLE.with(|slot| {
        let previous = slot.replace(Some(handle as usize));
        struct Restore<'a>(&'a Cell<Option<usize>>, Option<usize>);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _restore = Restore(slot, previous);
        f()
    })
}

pub(crate) fn is_factory_handle(handle: *mut c_void) -> bool {
    FACTORY_HANDLE.with(|slot| slot.get() == Some(handle as usize))
}

fn with_callback_impl<R>(
    handle: *mut c_void,
    self_ptr: *mut c_void,
    generation: u64,
    f: impl FnOnce() -> R,
) -> R {
    CALLBACK_IMPL.with(|slot| {
        let previous = slot.replace(Some((handle as usize, self_ptr as usize, generation)));
        struct Restore<'a>(
            &'a Cell<Option<(usize, usize, u64)>>,
            Option<(usize, usize, u64)>,
        );
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _restore = Restore(slot, previous);
        f()
    })
}

pub(crate) fn callback_impl(handle: *mut c_void) -> Option<*mut c_void> {
    CALLBACK_IMPL.with(|slot| {
        slot.get().and_then(|(active_handle, self_ptr, _)| {
            (active_handle == handle as usize).then_some(self_ptr as *mut c_void)
        })
    })
}

pub(crate) fn callback_generation() -> Option<u64> {
    CALLBACK_IMPL.with(|slot| slot.get().map(|(_, _, generation)| generation))
}

impl Drop for FfiNode {
    fn drop(&mut self) {
        // Deliberately touches nothing but the C side: the handle may itself be
        // mid-teardown here. Whoever abandons an unpublished generation clears
        // the pending slot that pointed at this state.
        if let Some(destroy) = self.vtable.destroy {
            self.call(|| destroy(self.handle));
        }
    }
}

impl Node for FfiNode {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        if self.vtable.process.is_some() {
            NodeKind::Blocking
        } else {
            NodeKind::Poll
        }
    }
    fn start(&self) {
        if let Some(f) = self.vtable.start {
            self.call(|| f(self.handle));
        }
    }
    fn set_generation(&self, generation: u64) {
        self.generation.store(generation, Ordering::Release);
    }
    fn stop(&self) {
        if let Some(f) = self.vtable.stop {
            self.call(|| f(self.handle));
        }
    }
    fn process(&self) -> Blocked {
        self.process_result().unwrap_or(Blocked::Done)
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Tick {
        self.poll_result().unwrap_or(Tick::Done)
    }
    fn take_body(self: Arc<Self>) -> NodeBody {
        match self.kind() {
            NodeKind::Blocking => NodeBody::Blocking(Box::new(move || self.process_result())),
            NodeKind::Poll => NodeBody::Poll(Box::new(move |_ctx| self.poll_result())),
            NodeKind::Async => unreachable!("C vtables do not define async nodes"),
        }
    }
    fn query_interface(&self, iface: AvpInterfaceId) -> Option<*const c_void> {
        let f = self.vtable.query_interface?;
        let p = self.call(|| f(self.handle, iface as u32));
        if p.is_null() { None } else { Some(p) }
    }
}

pub type VtableNode = FfiNode;
