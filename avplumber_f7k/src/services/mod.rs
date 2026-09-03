//! Services substrate.

pub mod clock;
pub mod correction;
pub mod timeline;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use crate::graph::AvpServiceId;
use crate::services::clock::ClockService;
use crate::services::correction::CorrectionService;
use crate::services::timeline::TimelineService;

pub struct ServiceRegistry {
    vtables: Mutex<HashMap<AvpServiceId, *const c_void>>,
    typed: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
    pub clocks: ClockService,
    pub corrections: CorrectionService,
    pub timelines: TimelineService,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self {
            vtables: Mutex::new(HashMap::new()),
            typed: Mutex::new(HashMap::new()),
            clocks: ClockService::new(),
            corrections: CorrectionService::new(),
            timelines: TimelineService::new(),
        }
    }

    pub fn register_vtable(&self, id: AvpServiceId, vtable: *const c_void) {
        self.vtables.lock().unwrap().insert(id, vtable);
    }

    pub fn query_vtable(&self, id: AvpServiceId) -> *const c_void {
        *self
            .vtables
            .lock()
            .unwrap()
            .get(&id)
            .unwrap_or(&std::ptr::null())
    }

    pub fn insert_typed<T: Any + Send + Sync>(&self, value: Arc<T>) {
        self.typed.lock().unwrap().insert(TypeId::of::<T>(), value);
    }

    pub fn get_typed<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.typed
            .lock()
            .unwrap()
            .get(&TypeId::of::<T>())?
            .clone()
            .downcast::<T>()
            .ok()
    }

    pub fn register(&mut self, id: AvpServiceId, vtable: *const c_void) {
        self.register_vtable(id, vtable);
    }

    pub fn query(&self, id: AvpServiceId) -> *const c_void {
        self.query_vtable(id)
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
