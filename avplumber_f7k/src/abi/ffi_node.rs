//! C node vtable wrapped as a native `Node`.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::capability::AvpInterfaceId;
use crate::graph::node::{Blocked, Node, NodeKind, Tick};
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

    fn flow_blocked(code: i32) -> Blocked {
        match code {
            3 | 4 => Blocked::Done,
            _ => Blocked::Again,
        }
    }
    fn flow_tick(code: i32) -> Tick {
        match code {
            2 => Tick::Idle,
            3 | 4 => Tick::Done,
            _ => Tick::Again,
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
        match self.vtable.process {
            Some(f) => Self::flow_blocked(self.call(|| f(self.handle))),
            None => Blocked::Done,
        }
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Tick {
        match self.vtable.poll {
            Some(f) => Self::flow_tick(self.call(|| f(self.handle))),
            None => Tick::Done,
        }
    }
    fn query_interface(&self, iface: AvpInterfaceId) -> Option<*const c_void> {
        let f = self.vtable.query_interface?;
        let p = self.call(|| f(self.handle, iface as u32));
        if p.is_null() { None } else { Some(p) }
    }
}

pub type VtableNode = FfiNode;
