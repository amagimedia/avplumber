//! Node factory registry: type name → constructor, with JSON params erased
//! through `NodeSpec` / `BuiltNode`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::Instance;
use crate::exec::ExecCtxId;
use crate::graph::{Node, NodeKind, NodePads};

thread_local! {
    static BUILD_GENERATION: Cell<Option<u64>> = const { Cell::new(None) };
}

pub(crate) fn with_build_generation<R>(generation: u64, f: impl FnOnce() -> R) -> R {
    BUILD_GENERATION.with(|slot| {
        let previous = slot.replace(Some(generation));
        struct Restore<'a>(&'a Cell<Option<u64>>, Option<u64>);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _restore = Restore(slot, previous);
        f()
    })
}

pub(crate) fn build_generation() -> Option<u64> {
    BUILD_GENERATION.with(Cell::get)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartPolicy {
    Off,
    RestartGroup,
    Panic,
    Exit,
}

impl RestartPolicy {
    pub fn parse_auto_restart(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "group" | "restart_group" => Ok(Self::RestartGroup),
            "panic" => Ok(Self::Panic),
            "exit" => Ok(Self::Exit),
            "on" | "restart_node" => Err(
                "auto_restart=restart_node is unsupported; isolated node restart is deferred"
                    .into(),
            ),
            other => Err(format!("invalid auto_restart value `{other}`")),
        }
    }

    pub fn parse_on_error(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "group" | "restart_group" => Ok(Self::RestartGroup),
            "panic" => Ok(Self::Panic),
            "exit" => Ok(Self::Exit),
            "on" | "restart_node" => Err(
                "on_error=restart_node is unsupported; isolated node restart is deferred".into(),
            ),
            other => Err(format!("invalid on_error value `{other}`")),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PlacementRequest {
    Blocking,
    EventLoop { name: Option<String> },
    TickSource { name: String },
}

impl PlacementRequest {
    pub fn to_exec_ctx(&self, service_hint: Option<&str>) -> ExecCtxId {
        match self {
            PlacementRequest::Blocking => ExecCtxId::Blocking,
            PlacementRequest::TickSource { name } => ExecCtxId::TickSource { name: name.clone() },
            PlacementRequest::EventLoop { name: Some(n) } => {
                ExecCtxId::EventLoop { name: n.clone() }
            }
            PlacementRequest::EventLoop { name: None } => ExecCtxId::EventLoop {
                name: service_hint.unwrap_or("default").to_string(),
            },
        }
    }
}

pub struct BuildCtx<'a> {
    pub instance: &'a Instance,
    pub name: &'a str,
    pub params: &'a Value,
}

impl<'a> BuildCtx<'a> {
    pub fn clock(&self, name: &str) -> Arc<dyn crate::services::clock::SyncGroup> {
        self.instance.services.clocks.get_or_create(name)
    }
    pub fn correction(&self, name: &str) -> Arc<crate::services::correction::CorrectionGroup> {
        self.instance.services.corrections.get_or_create(name)
    }
    pub fn timeline(&self, name: &str) -> Arc<crate::services::timeline::InMemoryTimeline> {
        self.instance.services.timelines.get_or_create(name)
    }
}

pub struct BuiltNode {
    pub node: Arc<dyn Node>,
    pub pads: NodePads,
    pub placement: PlacementRequest,
    pub sync_group: Option<String>,
    pub correction_group: Option<String>,
    pub restart: RestartPolicy,
    pub on_error: Option<RestartPolicy>,
    /// Edges the node names itself, from [`NodeSpec::bindings`]. Applied once by
    /// `create_node`; a reconstruction rebinds from the graph links those
    /// bindings created, so this field is not consulted again.
    pub bindings: Vec<(crate::core::PadDirection, String, String)>,
}

impl BuiltNode {
    pub fn from_node(node: Arc<dyn Node>) -> Self {
        let placement = match node.kind() {
            NodeKind::Blocking => PlacementRequest::Blocking,
            NodeKind::Poll | NodeKind::Async => PlacementRequest::EventLoop { name: None },
        };
        Self {
            pads: node.pads(),
            node,
            placement,
            sync_group: None,
            correction_group: None,
            restart: RestartPolicy::Off,
            on_error: None,
            bindings: Vec::new(),
        }
    }
}

/// `(instance_name, json_params) -> Arc<dyn Node>`.
///
/// Closed over whatever the registrant has; no `BuildCtx`, so it cannot look up
/// clocks/corrections/timelines by name at construction. Placement, pads, and
/// restart are recovered later by `BuiltNode::from_node` from `Node::kind` /
/// `Node::pads` (service-group hints stay `None`, restart stays `Off`). Use
/// `register()` for stubs and C-vtable factories that only produce an impl.
pub type NodeFactoryFn = Arc<dyn Fn(&str, &str) -> Result<Arc<dyn Node>, String> + Send + Sync>;

/// `(instance_name, json_params, &BuildCtx) -> BuiltNode`.
///
/// The native path: `BuildCtx` is the factory's handle onto `Instance` services,
/// and the return value carries pads/placement/restart/sync/correction instead
/// of making `create_node` infer them. `register_spec` installs this shape;
/// `resolve` prefers it over `NodeFactoryFn` when both exist for a type.
pub type BuiltFactoryFn =
    Arc<dyn for<'a> Fn(&str, &str, &BuildCtx<'a>) -> Result<BuiltNode, String> + Send + Sync>;
pub type ModuleInitFn = Box<dyn FnOnce(&mut FactoryRegistry) + Send>;

#[derive(Clone)]
pub(crate) enum ResolvedFactory {
    Legacy(NodeFactoryFn),
    Built(BuiltFactoryFn),
}

/// Persistent description of how to construct one logical node.
///
/// The active `Node` is generation-specific; this record deliberately keeps
/// the resolved factory and canonical parameters instead of retaining a
/// consumed `NodeBody`.
#[derive(Clone)]
pub struct NodeBlueprint {
    pub type_name: String,
    pub name: String,
    pub canonical_params: String,
    pub placement: ExecCtxId,
    pub pads: NodePads,
    pub kind: NodeKind,
    pub restart: RestartPolicy,
    pub on_error: Option<RestartPolicy>,
    pub sync_group: Option<String>,
    pub correction_group: Option<String>,
    pub service_hint: Option<String>,
    pub(crate) recipe: ConstructionRecipe,
}

#[derive(Clone)]
pub(crate) struct ConstructionRecipe {
    pub factory: ResolvedFactory,
    pub requested_placement: Option<PlacementRequest>,
    pub requested_restart: Option<RestartPolicy>,
    pub requested_on_error: Option<RestartPolicy>,
    pub requested_sync_group: Option<String>,
    pub requested_correction_group: Option<String>,
}

impl NodeBlueprint {
    pub(crate) fn build(&self, inst: &Instance) -> Result<BuiltNode, String> {
        self.recipe
            .factory
            .build(inst, &self.name, &self.canonical_params)
    }
}

impl ResolvedFactory {
    pub(crate) fn build(
        &self,
        inst: &Instance,
        instance_name: &str,
        params: &str,
    ) -> Result<BuiltNode, String> {
        match self {
            ResolvedFactory::Legacy(factory) => {
                Ok(BuiltNode::from_node(factory(instance_name, params)?))
            }
            ResolvedFactory::Built(factory) => {
                let parsed: Value = serde_json::from_str(params).map_err(|e| e.to_string())?;
                let ctx = BuildCtx {
                    instance: inst,
                    name: instance_name,
                    params: &parsed,
                };
                factory(instance_name, params, &ctx)
            }
        }
    }
}

pub trait NodeSpec: DeserializeOwned {
    const TYPE_NAME: &'static str;
    type Node: Node + 'static;
    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String>;

    /// Edges this node names in its own parameters, as
    /// `(direction, pad, edge_name)`. `demux` derives them from `routing`, so a
    /// script does not repeat the edge names in `src`/`dst`; everything else
    /// keeps the default and is bound by the envelope.
    fn bindings(&self) -> Vec<(crate::core::PadDirection, String, String)> {
        Vec::new()
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct NodeEnvelope {
    #[serde(rename = "type")]
    pub type_name: String,
    pub name: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub auto_restart: Option<String>,
    #[serde(default)]
    pub on_error: Option<String>,
    #[serde(default)]
    pub event_loop: Option<String>,
    #[serde(default)]
    pub tick_source: Option<String>,
    #[serde(default)]
    pub sync_group: Option<String>,
    #[serde(default)]
    pub correction_group: Option<String>,
    #[serde(default)]
    pub src: Option<Value>,
    #[serde(default)]
    pub dst: Option<Value>,
}

impl NodeEnvelope {
    pub fn extract(obj: &mut serde_json::Map<String, Value>) -> Result<Self, String> {
        let type_name = obj
            .remove("type")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| "missing type".to_string())?;
        let name = obj
            .remove("name")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| "missing name".to_string())?;
        let group = obj
            .remove("group")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let auto_restart = match obj.remove("auto_restart") {
            Some(Value::String(s)) => Some(s),
            Some(Value::Bool(true)) => {
                return Err(
                    "auto_restart=true requests restart_node; restart_node is unsupported, use \"group\""
                        .into(),
                );
            }
            Some(Value::Bool(false)) | None => None,
            Some(_) => return Err("auto_restart must be a string or boolean".into()),
        };
        let on_error = match obj.remove("on_error") {
            Some(Value::String(s)) => Some(s),
            None => None,
            Some(_) => return Err("on_error must be a string".into()),
        };
        let event_loop = obj
            .remove("event_loop")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let tick_source = obj
            .remove("tick_source")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let sync_group = obj
            .remove("sync_group")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let correction_group = obj
            .remove("correction_group")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let src = obj.remove("src");
        let dst = obj.remove("dst");
        Ok(Self {
            type_name,
            name,
            group,
            auto_restart,
            on_error,
            event_loop,
            tick_source,
            sync_group,
            correction_group,
            src,
            dst,
        })
    }

    pub fn restart(&self) -> Result<RestartPolicy, String> {
        self.auto_restart
            .as_deref()
            .map(RestartPolicy::parse_auto_restart)
            .unwrap_or(Ok(RestartPolicy::Off))
    }

    pub fn on_error(&self) -> Result<Option<RestartPolicy>, String> {
        self.on_error
            .as_deref()
            .map(RestartPolicy::parse_on_error)
            .transpose()
    }

    pub fn placement(&self, kind: NodeKind) -> Result<PlacementRequest, String> {
        if kind.is_blocking() {
            return Ok(PlacementRequest::Blocking);
        }
        match (self.event_loop.as_deref(), self.tick_source.as_deref()) {
            (Some(_), Some(_)) => Err("event_loop and tick_source are mutually exclusive".into()),
            (_, Some(ts)) => Ok(PlacementRequest::TickSource {
                name: ts.to_string(),
            }),
            (Some(el), None) => Ok(PlacementRequest::EventLoop {
                name: Some(el.to_string()),
            }),
            (None, None) => Ok(PlacementRequest::EventLoop { name: None }),
        }
    }
}

pub struct FactoryRegistry {
    factories: HashMap<String, NodeFactoryFn>,
    built: HashMap<String, BuiltFactoryFn>,
    module_inits: Vec<ModuleInitFn>,
}

impl FactoryRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            built: HashMap::new(),
            module_inits: Vec::new(),
        }
    }

    pub fn register<F>(&mut self, type_name: &str, f: F)
    where
        F: Fn(&str, &str) -> Result<Arc<dyn Node>, String> + Send + Sync + 'static,
    {
        self.factories.insert(type_name.to_string(), Arc::new(f));
    }

    pub fn register_spec<S: NodeSpec + 'static>(&mut self) {
        self.built.insert(
            S::TYPE_NAME.to_string(),
            Arc::new(|name, params, ctx| {
                let spec: S = serde_json::from_str(params).map_err(|e| e.to_string())?;
                let bindings = spec.bindings();
                let mut built = BuiltNode::from_node(Arc::new(spec.build(name, ctx)?));
                built.bindings = bindings;
                Ok(built)
            }),
        );
    }

    pub fn register_module_init(&mut self, f: ModuleInitFn) {
        self.module_inits.push(f);
    }

    pub fn run_module_inits(&mut self) {
        let inits = std::mem::take(&mut self.module_inits);
        for f in inits {
            f(self);
        }
    }

    pub fn create(
        &self,
        type_name: &str,
        instance_name: &str,
        params: &str,
    ) -> Result<Arc<dyn Node>, String> {
        if let Some(f) = self.factories.get(type_name) {
            return f(instance_name, params);
        }
        Err(format!("unknown node type: {type_name}"))
    }

    pub fn create_built(
        &self,
        inst: &Instance,
        type_name: &str,
        instance_name: &str,
        params: &str,
    ) -> Result<BuiltNode, String> {
        self.resolve(type_name)
            .ok_or_else(|| format!("unknown node type: {type_name}"))?
            .build(inst, instance_name, params)
    }

    pub(crate) fn resolve(&self, type_name: &str) -> Option<ResolvedFactory> {
        self.built
            .get(type_name)
            .cloned()
            .map(ResolvedFactory::Built)
            .or_else(|| {
                self.factories
                    .get(type_name)
                    .cloned()
                    .map(ResolvedFactory::Legacy)
            })
    }

    pub fn types(&self) -> impl Iterator<Item = &str> {
        self.factories
            .keys()
            .chain(self.built.keys())
            .map(String::as_str)
    }
}

impl Default for FactoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}
