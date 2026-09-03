//! avplumber_f7k — Rust core for avplumber.

pub mod abi;
pub mod control;
pub mod core;
pub mod exec;
pub mod factory;
pub mod graph;
pub mod scaffold;
pub mod services;
pub mod supervisor;

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

pub use abi::{AvpBuffer, AvpNodeVtable, AvpSpec, EdgeCoupling, FfiNode, VtableNode};
pub use core::{CoreError, EdgeInstance, NodeInstance, NodeRequest, PadDirection};
pub use exec::{
    AsyncExecutor, BlockingExecutor, ExecCtxId, Executor, ExecutorState, Generation, NodeOutcome,
    OutcomeReporter,
};
pub use factory::{
    BuildCtx, BuiltNode, FactoryRegistry, NodeBlueprint, NodeEnvelope, NodeSpec, PlacementRequest,
    RestartPolicy,
};
pub use graph::{
    AVP_NOPTS, AvpInterfaceId, AvpMediaType, AvpMediaVtable, AvpRational, AvpServiceId, Blocked,
    BufferedEdge, ChannelLayout, DirectEdge, Edge, EdgeEvent, EdgeItem, EdgeKind, EdgeLink,
    EdgeRestart, EdgeWaker, Graph, Media, Node, NodeBody, NodeError, NodeKind, NodePads, NodePhase,
    NodePollContext, PacketSpec, Pad, PadDecl, Push, Spec, Tick, Ts, Vertex, Wakeup,
    generation_reader, generation_writer,
};
pub use scaffold::{SisoAdapter, SisoAsyncAdapter, SisoNode, SisoPollAdapter};
pub use services::ServiceRegistry;
pub use supervisor::{
    Group, GroupState, GroupStatus, Reconstruction, RestartHook, RestartRequest, SupervisorAction,
    SupervisorActionHook,
};

pub struct Instance {
    pub(crate) inner: Arc<InstanceInner>,
}

#[doc(hidden)]
pub struct InstanceInner {
    pub(crate) graph: Arc<Mutex<Graph>>,
    pub(crate) factories: Mutex<FactoryRegistry>,
    pub(crate) services: ServiceRegistry,
    pub(crate) media_vtables: Mutex<HashMap<AvpMediaType, AvpMediaVtable>>,
    pub(crate) shared: Mutex<HashMap<(String, String), SharedEntry>>,
    pub(crate) groups: Mutex<HashMap<String, Arc<Group>>>,
    pub(crate) nodes: Mutex<HashMap<String, NodeInstance>>,
    pub(crate) group_handles: Mutex<HashMap<String, Box<crate::abi::AvpGroup>>>,
    pub(crate) node_handles: Mutex<HashMap<String, Box<crate::abi::AvpNode>>>,
    pub(crate) pending_node_handles: Mutex<HashMap<String, Box<crate::abi::AvpNode>>>,
    pub(crate) edge_handles: Mutex<HashMap<String, Box<crate::abi::AvpEdge>>>,
    pub(crate) edges: Mutex<HashMap<String, crate::core::EdgeRecord>>,
    pub(crate) planned_capacity: Mutex<HashMap<String, usize>>,
    pub(crate) supervisor_action_hook: Mutex<Option<crate::supervisor::SupervisorActionHook>>,
}

// All mutable native state is behind locks. ABI pointers are opaque identities
// whose pointees are owned by the embedding Instance and remain valid until its
// groups have stopped; reconstruction is serialized by group manager threads.
unsafe impl Send for InstanceInner {}
unsafe impl Sync for InstanceInner {}

impl std::ops::Deref for Instance {
    type Target = InstanceInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub(crate) struct SharedEntry {
    pub obj: *mut c_void,
    pub vtable: Option<AvpMediaVtable>,
}

unsafe impl Send for SharedEntry {}

impl Drop for SharedEntry {
    fn drop(&mut self) {
        if let Some(vt) = self.vtable {
            if !self.obj.is_null() {
                (vt.release)(self.obj);
            }
        }
    }
}

impl Instance {
    pub fn new() -> Self {
        let graph = Arc::new(Mutex::new(Graph::new()));
        let mut factories = FactoryRegistry::new();
        factories.run_module_inits();
        Self {
            inner: Arc::new(InstanceInner {
                graph,
                factories: Mutex::new(factories),
                services: ServiceRegistry::new(),
                media_vtables: Mutex::new(HashMap::new()),
                shared: Mutex::new(HashMap::new()),
                groups: Mutex::new(HashMap::new()),
                group_handles: Mutex::new(HashMap::new()),
                node_handles: Mutex::new(HashMap::new()),
                pending_node_handles: Mutex::new(HashMap::new()),
                edge_handles: Mutex::new(HashMap::new()),
                nodes: Mutex::new(HashMap::new()),
                edges: Mutex::new(HashMap::new()),
                planned_capacity: Mutex::new(HashMap::new()),
                supervisor_action_hook: Mutex::new(None),
            }),
        }
    }

    pub(crate) fn from_inner(inner: Arc<InstanceInner>) -> Self {
        Self { inner }
    }

    pub(crate) fn shutdown(&self) {
        let groups = self
            .groups
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for group in &groups {
            group.stop();
        }
        for group in groups {
            group.join_reconstruction_workers();
        }
    }

    /// Replaces process-level panic/exit behavior, primarily for embedders and
    /// tests that need to observe escalation without terminating the process.
    pub fn set_supervisor_action_hook(&self, hook: crate::supervisor::SupervisorActionHook) {
        *self.supervisor_action_hook.lock().unwrap() = Some(hook);
    }

    pub fn plan_capacity(&self, name: &str, cap: usize) {
        self.planned_capacity
            .lock()
            .unwrap()
            .insert(name.to_string(), cap);
    }

    pub fn named_edge(&self, name: &str) -> Arc<dyn Edge> {
        self.get_or_create_edge(name).edge
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        let groups = self
            .groups
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for group in groups {
            group.stop();
        }
    }
}

pub fn register_factory<F>(inst: &Instance, type_name: &str, f: F)
where
    F: Fn(&str, &str) -> Result<Arc<dyn Node>, String> + Send + Sync + 'static,
{
    inst.factories.lock().unwrap().register(type_name, f);
}

pub fn register_spec<S: NodeSpec + 'static>(inst: &Instance) {
    inst.factories.lock().unwrap().register_spec::<S>();
}
