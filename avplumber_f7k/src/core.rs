//! Authoritative native graph-management API.
//!
//! Control and embedding code call this layer. `abi/` only translates C
//! strings, pointers, and handles into these operations.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use crate::exec::ExecCtxId;
use crate::factory::{
    ConstructionRecipe, NodeBlueprint, NodeEnvelope, PlacementRequest, RestartPolicy,
};
use crate::graph::{
    BufferedEdge, DirectEdge, Edge, EdgeKind, EdgeLink, Graph, Node, NodeKind, NodePads, Vertex,
    check_pad_media, generation_reader, generation_writer,
};
use crate::{Group, Instance};

#[derive(Debug)]
pub enum CoreError {
    AlreadyExists { kind: &'static str, name: String },
    NotFound { kind: &'static str, name: String },
    Invalid(String),
    Operation(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::AlreadyExists { kind, name } => write!(f, "{kind} name busy: {name}"),
            CoreError::NotFound { kind, name } => write!(f, "unknown {kind}: {name}"),
            CoreError::Invalid(message) | CoreError::Operation(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CoreError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadDirection {
    Input,
    Output,
}

#[derive(Clone, Debug)]
pub struct NodeRequest {
    pub type_name: String,
    pub name: String,
    pub params: Value,
    pub placement: Option<PlacementRequest>,
    pub restart: Option<RestartPolicy>,
    pub on_error: Option<RestartPolicy>,
    pub sync_group: Option<String>,
    pub correction_group: Option<String>,
}

impl NodeRequest {
    pub fn new(type_name: impl Into<String>, name: impl Into<String>, params: Value) -> Self {
        Self {
            type_name: type_name.into(),
            name: name.into(),
            params,
            placement: None,
            restart: None,
            on_error: None,
            sync_group: None,
            correction_group: None,
        }
    }

    /// Parse framework keys from the parameter object used by the C ABI.
    pub fn from_json(
        type_name: impl Into<String>,
        name: impl Into<String>,
        params_json: &str,
    ) -> Result<Self, CoreError> {
        let type_name = type_name.into();
        let name = name.into();
        let params: Value =
            serde_json::from_str(params_json).map_err(|e| CoreError::Invalid(e.to_string()))?;
        let mut obj = params
            .as_object()
            .cloned()
            .ok_or_else(|| CoreError::Invalid("node params must be a JSON object".into()))?;
        obj.insert("type".into(), Value::String(type_name));
        obj.insert("name".into(), Value::String(name));
        let envelope = NodeEnvelope::extract(&mut obj).map_err(CoreError::Invalid)?;
        if let Some(group) = envelope.group.as_ref() {
            obj.insert("group".into(), Value::String(group.clone()));
        }
        if let Some(src) = envelope.src.as_ref() {
            obj.insert("src".into(), src.clone());
        }
        if let Some(dst) = envelope.dst.as_ref() {
            obj.insert("dst".into(), dst.clone());
        }
        Self::from_envelope(envelope, Value::Object(obj))
    }

    pub fn from_envelope(envelope: NodeEnvelope, params: Value) -> Result<Self, CoreError> {
        let restart = envelope
            .auto_restart
            .as_deref()
            .map(RestartPolicy::parse_auto_restart)
            .transpose()
            .map_err(CoreError::Invalid)?;
        let on_error = envelope
            .on_error
            .as_deref()
            .map(RestartPolicy::parse_on_error)
            .transpose()
            .map_err(CoreError::Invalid)?;
        let placement = match (
            envelope.event_loop.as_deref(),
            envelope.tick_source.as_deref(),
        ) {
            (Some(_), Some(_)) => {
                return Err(CoreError::Invalid(
                    "event_loop and tick_source are mutually exclusive".into(),
                ));
            }
            (_, Some(name)) => Some(PlacementRequest::TickSource {
                name: name.to_string(),
            }),
            (Some(name), None) => Some(PlacementRequest::EventLoop {
                name: Some(name.to_string()),
            }),
            (None, None) => None,
        };
        Ok(Self {
            type_name: envelope.type_name,
            name: envelope.name,
            params,
            placement,
            restart,
            on_error,
            sync_group: envelope.sync_group,
            correction_group: envelope.correction_group,
        })
    }
}

#[derive(Clone)]
pub struct NodeInstance {
    pub name: String,
    pub node: Arc<dyn Node>,
    pub pads: NodePads,
    pub exec_ctx: ExecCtxId,
    pub restart: RestartPolicy,
    pub on_error: Option<RestartPolicy>,
    pub sync_group: Option<String>,
    pub correction_group: Option<String>,
    pub service_hint: Option<String>,
    pub blueprint: Arc<NodeBlueprint>,
    pub active_generation: u64,
}

#[derive(Clone)]
pub struct EdgeInstance {
    pub name: String,
    pub edge: Arc<dyn Edge>,
}

pub(crate) struct EdgeRecord {
    pub edge: Arc<dyn Edge>,
    producer: Option<(String, String)>,
    consumer: Option<(String, String)>,
    linked: bool,
}

impl EdgeRecord {
    fn new(edge: Arc<dyn Edge>) -> Self {
        Self {
            edge,
            producer: None,
            consumer: None,
            linked: false,
        }
    }
}

fn service_hint(correction_group: Option<String>, sync_group: Option<String>) -> Option<String> {
    correction_group.or(sync_group)
}

fn placement_for(
    kind: NodeKind,
    requested: Option<PlacementRequest>,
    built: PlacementRequest,
    hint: Option<&str>,
) -> ExecCtxId {
    if kind.is_blocking() {
        return ExecCtxId::Blocking;
    }
    requested.unwrap_or(built).to_exec_ctx(hint)
}

struct EffectiveNode {
    node: Arc<dyn Node>,
    pads: NodePads,
    kind: NodeKind,
    exec_ctx: ExecCtxId,
    restart: RestartPolicy,
    on_error: Option<RestartPolicy>,
    sync_group: Option<String>,
    correction_group: Option<String>,
    service_hint: Option<String>,
}

fn resolve_effective_node(
    mut built: crate::factory::BuiltNode,
    recipe: &ConstructionRecipe,
) -> EffectiveNode {
    let correction_group = recipe
        .requested_correction_group
        .clone()
        .or(built.correction_group.take());
    let sync_group = recipe
        .requested_sync_group
        .clone()
        .or(built.sync_group.take());
    let service_hint = service_hint(correction_group.clone(), sync_group.clone());
    let restart = recipe.requested_restart.unwrap_or(built.restart);
    let on_error = recipe.requested_on_error.or(built.on_error);
    let kind = built.node.kind();
    let exec_ctx = placement_for(
        kind,
        recipe.requested_placement.clone(),
        built.placement,
        service_hint.as_deref(),
    );
    EffectiveNode {
        node: built.node,
        pads: built.pads,
        kind,
        exec_ctx,
        restart,
        on_error,
        sync_group,
        correction_group,
        service_hint,
    }
}

fn validate_replacement(
    current: &NodeInstance,
    replacement: &EffectiveNode,
) -> Result<(), CoreError> {
    let blueprint = &current.blueprint;
    if replacement.node.name() != blueprint.name {
        return Err(CoreError::Invalid(format!(
            "replacement for `{}` returned node named `{}`",
            blueprint.name,
            replacement.node.name()
        )));
    }
    if replacement.kind != blueprint.kind {
        return Err(CoreError::Invalid(format!(
            "replacement for `{}` changed kind from {:?} to {:?}",
            blueprint.name, blueprint.kind, replacement.kind
        )));
    }
    if replacement.pads != blueprint.pads {
        return Err(CoreError::Invalid(format!(
            "replacement for `{}` changed its pad declaration",
            blueprint.name
        )));
    }
    if replacement.exec_ctx != blueprint.placement {
        return Err(CoreError::Invalid(format!(
            "replacement for `{}` changed placement from {:?} to {:?}",
            blueprint.name, blueprint.placement, replacement.exec_ctx
        )));
    }
    if replacement.restart != blueprint.restart
        || replacement.on_error != blueprint.on_error
        || replacement.sync_group != blueprint.sync_group
        || replacement.correction_group != blueprint.correction_group
        || replacement.service_hint != blueprint.service_hint
    {
        return Err(CoreError::Invalid(format!(
            "replacement for `{}` changed effective policy metadata",
            blueprint.name
        )));
    }
    Ok(())
}

fn validate_vertex_topology(vertex: &Vertex, blueprint: &NodeBlueprint) -> Result<(), CoreError> {
    let source_media = blueprint
        .pads
        .sources
        .iter()
        .map(|pad| (pad.name.clone(), pad.media))
        .collect::<HashMap<_, _>>();
    let sink_media = blueprint
        .pads
        .sinks
        .iter()
        .map(|pad| (pad.name.clone(), pad.media))
        .collect::<HashMap<_, _>>();
    if vertex.source_media != source_media || vertex.sink_media != sink_media {
        return Err(CoreError::Invalid(format!(
            "node `{}` topology no longer matches its blueprint pads",
            blueprint.name
        )));
    }
    if vertex
        .sources
        .keys()
        .any(|pad| !source_media.contains_key(pad))
        || vertex.sinks.keys().any(|pad| !sink_media.contains_key(pad))
    {
        return Err(CoreError::Invalid(format!(
            "node `{}` has an edge bound to an undeclared pad",
            blueprint.name
        )));
    }
    Ok(())
}

fn edge_maps_identical(
    current: &HashMap<String, Arc<dyn Edge>>,
    expected: &HashMap<String, Arc<dyn Edge>>,
) -> bool {
    current.len() == expected.len()
        && current.iter().all(|(pad, edge)| {
            expected
                .get(pad)
                .is_some_and(|previous| Arc::ptr_eq(edge, previous))
        })
}

fn fence_group_edges(graph: &Graph, members: &[String], old_generation: u64, new_generation: u64) {
    let members = members.iter().map(String::as_str).collect::<HashSet<_>>();
    for link in graph.links() {
        let producer_inside = members.contains(link.producer.as_str());
        let consumer_inside = members.contains(link.consumer.as_str());
        let kind = match (producer_inside, consumer_inside) {
            (true, true) => Some(crate::graph::EdgeRestart::Internal),
            (true, false) => Some(crate::graph::EdgeRestart::Egress),
            (false, true) => Some(crate::graph::EdgeRestart::Ingress),
            (false, false) => None,
        };
        if let Some(kind) = kind {
            link.edge.restart(old_generation, new_generation, kind);
        }
    }
}

fn pad_media(
    vertex: &Vertex,
    direction: PadDirection,
    pad: &str,
) -> Result<Option<crate::graph::AvpMediaType>, CoreError> {
    let declared = match direction {
        PadDirection::Input => &vertex.source_media,
        PadDirection::Output => &vertex.sink_media,
    };
    if declared.is_empty() {
        return Ok(None);
    }
    declared.get(pad).copied().map(Some).ok_or_else(|| {
        CoreError::Invalid(format!(
            "node {} has no {direction:?} pad `{pad}`",
            vertex.name
        ))
    })
}

impl Instance {
    pub fn create_node(&self, request: NodeRequest) -> Result<NodeInstance, CoreError> {
        if self.nodes.lock().unwrap().contains_key(&request.name) {
            return Err(CoreError::AlreadyExists {
                kind: "node",
                name: request.name,
            });
        }

        let params_json = request.params.to_string();
        let factory = self
            .factories
            .lock()
            .unwrap()
            .resolve(&request.type_name)
            .ok_or_else(|| {
                CoreError::Operation(format!("unknown node type: {}", request.type_name))
            })?;
        let mut built = crate::factory::with_build_generation(1, || {
            factory.build(self, &request.name, &params_json)
        })
        .map_err(CoreError::Operation)?;
        let self_bindings = std::mem::take(&mut built.bindings);
        let recipe = ConstructionRecipe {
            factory,
            requested_placement: request.placement,
            requested_restart: request.restart,
            requested_on_error: request.on_error,
            requested_sync_group: request.sync_group,
            requested_correction_group: request.correction_group,
        };
        let effective = resolve_effective_node(built, &recipe);
        if effective.node.name() != request.name {
            return Err(CoreError::Invalid(format!(
                "factory for `{}` returned node named `{}`",
                request.name,
                effective.node.name()
            )));
        }
        let blueprint = Arc::new(NodeBlueprint {
            type_name: request.type_name,
            name: request.name.clone(),
            canonical_params: params_json,
            placement: effective.exec_ctx.clone(),
            pads: effective.pads.clone(),
            kind: effective.kind,
            restart: effective.restart,
            on_error: effective.on_error,
            sync_group: effective.sync_group.clone(),
            correction_group: effective.correction_group.clone(),
            service_hint: effective.service_hint.clone(),
            recipe,
        });
        let native = NodeInstance {
            name: request.name.clone(),
            node: effective.node.clone(),
            pads: effective.pads.clone(),
            exec_ctx: effective.exec_ctx,
            restart: effective.restart,
            on_error: effective.on_error,
            sync_group: effective.sync_group,
            correction_group: effective.correction_group,
            service_hint: effective.service_hint,
            blueprint,
            active_generation: 1,
        };

        self.graph
            .lock()
            .unwrap()
            .add_vertex(Vertex {
                name: request.name.clone(),
                node: effective.node,
                sources: HashMap::new(),
                sinks: HashMap::new(),
                source_media: native
                    .pads
                    .sources
                    .iter()
                    .map(|pad| (pad.name.clone(), pad.media))
                    .collect(),
                sink_media: native
                    .pads
                    .sinks
                    .iter()
                    .map(|pad| (pad.name.clone(), pad.media))
                    .collect(),
            })
            .map_err(CoreError::Operation)?;
        self.nodes
            .lock()
            .unwrap()
            .insert(request.name.clone(), native.clone());

        // Bindings the node declared itself go through the same `bind_edge` path
        // as an envelope's `src`/`dst`, now that the vertex exists. A failure
        // leaves no half-created node behind, like `node.add`'s own rollback.
        for (direction, pad, edge_name) in self_bindings {
            if let Err(error) = self.bind_edge(&request.name, &pad, direction, &edge_name) {
                let _ = self.destroy_node(&request.name);
                return Err(error);
            }
        }
        Ok(native)
    }

    pub fn node(&self, name: &str) -> Option<NodeInstance> {
        self.nodes.lock().unwrap().get(name).cloned()
    }

    pub fn group(&self, name: &str) -> Option<Arc<Group>> {
        self.groups.lock().unwrap().get(name).cloned()
    }

    pub(crate) fn replace_node_impl(
        &self,
        name: &str,
        node: Arc<dyn Node>,
        self_ptr: *mut std::ffi::c_void,
        vtable: &'static crate::abi::AvpNodeVtable,
    ) {
        let mut graph = self.graph.lock().unwrap();
        let mut nodes = self.nodes.lock().unwrap();
        let mut handles = self.node_handles.lock().unwrap();
        if let Some(vertex) = graph.vertex_mut(name) {
            vertex.node = node.clone();
        }
        if let Some(instance) = nodes.get_mut(name) {
            instance.node = node.clone();
        }
        if let Some(handle) = handles.get_mut(name) {
            handle.node = node;
            handle.self_ptr = self_ptr;
            handle.vtable = Some(vtable);
        }
    }

    /// Rebuild all members away from the graph, then publish the complete set
    /// under the graph/node-record locks. Failed construction or validation
    /// leaves every active node and edge binding untouched.
    ///
    /// Runs on the group's reconstruction worker thread, concurrently with the
    /// public API, so before publishing it rechecks both the topology it built
    /// against and the group's cancellation epoch: a `stop` meanwhile turns the
    /// whole transaction into a no-op.
    pub fn reconstruct_group(&self, group_name: &str) -> Result<(), CoreError> {
        let group = self.group(group_name).ok_or_else(|| CoreError::NotFound {
            kind: "group",
            name: group_name.to_string(),
        })?;
        let members = group.members();
        let result = self.rebuild_members(&group, &members);
        if result.is_err() {
            // Every replacement has been dropped by now, C destroy callbacks
            // included, so no stable handle is left holding their freed state.
            self.clear_pending_c_state(&members);
        }
        result
    }

    /// Nulls the C state an abandoned generation staged on the stable handles.
    /// Only meaningful after the abandoned nodes are gone.
    fn clear_pending_c_state(&self, members: &[String]) {
        let mut handles = self.node_handles.lock().unwrap();
        for name in members {
            if let Some(handle) = handles.get_mut(name) {
                handle.pending_self_ptr = std::ptr::null_mut();
                handle.pending_vtable = None;
                handle
                    .producer_leases
                    .append(&mut handle.pending_producer_leases);
                handle
                    .direct_reader_leases
                    .append(&mut handle.pending_direct_reader_leases);
            }
        }
    }

    fn rebuild_members(&self, group: &Group, members: &[String]) -> Result<(), CoreError> {
        let epoch = group.reconstruction_epoch();
        let next_generation = group.generation().saturating_add(1);
        let old = {
            let nodes = self.nodes.lock().unwrap();
            members
                .iter()
                .map(|name| {
                    nodes.get(name).cloned().ok_or_else(|| CoreError::NotFound {
                        kind: "node",
                        name: name.clone(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let replacements = crate::factory::with_build_generation(next_generation, || {
            old.iter()
                .map(|current| {
                    let built = current
                        .blueprint
                        .build(self)
                        .map_err(CoreError::Operation)?;
                    let effective = resolve_effective_node(built, &current.blueprint.recipe);
                    validate_replacement(current, &effective)?;
                    Ok(NodeInstance {
                        name: current.name.clone(),
                        node: effective.node,
                        pads: effective.pads,
                        exec_ctx: effective.exec_ctx,
                        restart: effective.restart,
                        on_error: effective.on_error,
                        sync_group: effective.sync_group,
                        correction_group: effective.correction_group,
                        service_hint: effective.service_hint,
                        blueprint: current.blueprint.clone(),
                        active_generation: next_generation,
                    })
                })
                .collect::<Result<Vec<_>, CoreError>>()
        })?;

        let bindings = {
            let graph = self.graph.lock().unwrap();
            old.iter()
                .map(|current| {
                    let vertex =
                        graph
                            .vertex(&current.name)
                            .ok_or_else(|| CoreError::NotFound {
                                kind: "node",
                                name: current.name.clone(),
                            })?;
                    if !Arc::ptr_eq(&vertex.node, &current.node) {
                        return Err(CoreError::Invalid(format!(
                            "node `{}` changed during reconstruction",
                            current.name
                        )));
                    }
                    validate_vertex_topology(vertex, &current.blueprint)?;
                    Ok((vertex.sources.clone(), vertex.sinks.clone()))
                })
                .collect::<Result<Vec<_>, CoreError>>()?
        };

        for (replacement, (sources, sinks)) in replacements.iter().zip(&bindings) {
            for (pad, edge) in sources {
                replacement
                    .node
                    .bind_source(pad, generation_reader(edge.clone(), next_generation));
                edge.rearm_spec();
            }
            for (pad, edge) in sinks {
                replacement
                    .node
                    .bind_sink(pad, generation_writer(edge.clone(), next_generation));
            }
        }

        // Kept out of the publish scope: on abort the replacements must run
        // their destructors, C destroy callbacks included, after the locks are
        // released.
        let mut abandoned = Some(replacements);
        let published = {
            let mut graph = self.graph.lock().unwrap();
            let mut nodes = self.nodes.lock().unwrap();
            let mut handles = self.node_handles.lock().unwrap();
            let mut publish = || -> Result<(), CoreError> {
                if group.reconstruction_epoch() != epoch {
                    return Err(CoreError::Invalid(format!(
                        "group `{}` reconstruction was cancelled before publish",
                        group.name()
                    )));
                }
                for (index, current) in old.iter().enumerate() {
                    let vertex =
                        graph
                            .vertex(&current.name)
                            .ok_or_else(|| CoreError::NotFound {
                                kind: "node",
                                name: current.name.clone(),
                            })?;
                    let record = nodes
                        .get(&current.name)
                        .ok_or_else(|| CoreError::NotFound {
                            kind: "node",
                            name: current.name.clone(),
                        })?;
                    if !Arc::ptr_eq(&vertex.node, &current.node)
                        || !Arc::ptr_eq(&record.node, &current.node)
                        || !edge_maps_identical(&vertex.sources, &bindings[index].0)
                        || !edge_maps_identical(&vertex.sinks, &bindings[index].1)
                    {
                        return Err(CoreError::Invalid(format!(
                            "node `{}` or its edge bindings changed during reconstruction",
                            current.name
                        )));
                    }
                }
                fence_group_edges(&graph, members, group.generation(), next_generation);
                for replacement in abandoned.take().expect("replacements built above") {
                    graph
                        .vertex_mut(&replacement.name)
                        .expect("replacement vertex validated")
                        .node = replacement.node.clone();
                    if let Some(handle) = handles.get_mut(&replacement.name) {
                        handle.node = replacement.node.clone();
                        handle.pads = replacement.pads.clone();
                        handle.exec_ctx = replacement.exec_ctx.clone();
                        handle.restart = replacement.restart;
                        handle.service_hint = replacement.service_hint.clone();
                        if let Some(vtable) = handle.pending_vtable.take() {
                            handle.self_ptr = handle.pending_self_ptr;
                            handle.pending_self_ptr = std::ptr::null_mut();
                            handle.vtable = Some(vtable);
                        }
                        let leases = std::mem::take(&mut handle.pending_producer_leases);
                        handle.producer_leases.extend(leases);
                        let leases = std::mem::take(&mut handle.pending_direct_reader_leases);
                        handle.direct_reader_leases.extend(leases);
                    }
                    nodes.insert(replacement.name.clone(), replacement);
                }
                graph.refresh_direct_bindings();
                Ok(())
            };
            publish()
        };
        // Past the guards, so an aborted transaction runs its C destroy
        // callbacks with no lock held.
        drop(abandoned);
        published
    }

    pub fn destroy_node(&self, name: &str) -> Result<(), CoreError> {
        if !self.nodes.lock().unwrap().contains_key(name) {
            return Err(CoreError::NotFound {
                kind: "node",
                name: name.to_string(),
            });
        }
        {
            let groups = self
                .groups
                .lock()
                .unwrap()
                .values()
                .cloned()
                .collect::<Vec<_>>();
            for group in &groups {
                if !group.members().iter().any(|member| member == name) {
                    continue;
                }
                if group.state() != crate::GroupState::Idle {
                    return Err(CoreError::Invalid(format!(
                        "cannot destroy node `{name}` while group `{}` is active",
                        group.name()
                    )));
                }
                // An idle group can still have a reconstruction worker
                // unwinding: it cannot publish any more, but it may be inside a
                // factory writing to this node's stable handle. Let it finish
                // before the records and handles go away.
                group.join_reconstruction_workers();
            }
            for group in groups {
                group.remove(name);
            }
        }
        let edge_names = self
            .edges
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, edge)| {
                edge.producer.as_ref().is_some_and(|(node, _)| node == name)
                    || edge.consumer.as_ref().is_some_and(|(node, _)| node == name)
            })
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        for edge_name in edge_names {
            self.destroy_edge(&edge_name)?;
            self.edge_handles.lock().unwrap().remove(&edge_name);
        }
        self.nodes.lock().unwrap().remove(name);
        self.graph.lock().unwrap().remove_vertex(name);
        self.pending_node_handles.lock().unwrap().remove(name);
        Ok(())
    }

    pub fn get_or_create_edge(&self, name: &str) -> EdgeInstance {
        let mut edges = self.edges.lock().unwrap();
        let edge = edges
            .entry(name.to_string())
            .or_insert_with(|| {
                let capacity = self
                    .planned_capacity
                    .lock()
                    .unwrap()
                    .get(name)
                    .copied()
                    .unwrap_or(0);
                EdgeRecord::new(Arc::new(BufferedEdge::new(capacity)))
            })
            .edge
            .clone();
        EdgeInstance {
            name: name.to_string(),
            edge,
        }
    }

    pub fn edge_link(&self, name: &str) -> Option<EdgeLink> {
        self.graph
            .lock()
            .unwrap()
            .links()
            .iter()
            .find(|link| link.name == name)
            .cloned()
    }

    pub fn bind_edge(
        &self,
        node_name: &str,
        pad: &str,
        direction: PadDirection,
        edge_name: &str,
    ) -> Result<EdgeInstance, CoreError> {
        let node_generation = self
            .node(node_name)
            .ok_or_else(|| CoreError::NotFound {
                kind: "node",
                name: node_name.to_string(),
            })?
            .active_generation;
        let mut graph = self.graph.lock().unwrap();
        let vertex = graph.vertex(node_name).ok_or_else(|| CoreError::NotFound {
            kind: "node",
            name: node_name.to_string(),
        })?;
        pad_media(vertex, direction, pad)?;
        let node = vertex.node.clone();

        let mut edges = self.edges.lock().unwrap();
        let record = edges.entry(edge_name.to_string()).or_insert_with(|| {
            let capacity = self
                .planned_capacity
                .lock()
                .unwrap()
                .get(edge_name)
                .copied()
                .unwrap_or(0);
            EdgeRecord::new(Arc::new(BufferedEdge::new(capacity)))
        });
        let endpoint = (node_name.to_string(), pad.to_string());
        let existing = match direction {
            PadDirection::Input => &record.consumer,
            PadDirection::Output => &record.producer,
        };
        if let Some(existing) = existing {
            if existing != &endpoint {
                return Err(CoreError::Invalid(format!(
                    "edge `{edge_name}` already has a {direction:?} endpoint"
                )));
            }
        } else {
            self.ensure_topology_idle(&[node_name])?;
        }

        let edge = record.edge.clone();
        let proposed_producer = match direction {
            PadDirection::Input => record.producer.clone(),
            PadDirection::Output => Some(endpoint.clone()),
        };
        let proposed_consumer = match direction {
            PadDirection::Input => Some(endpoint.clone()),
            PadDirection::Output => record.consumer.clone(),
        };
        let complete = match (&proposed_producer, &proposed_consumer) {
            (Some((producer, producer_pad)), Some((consumer, consumer_pad))) if !record.linked => {
                if graph.edge(edge_name).is_some() {
                    return Err(CoreError::AlreadyExists {
                        kind: "edge",
                        name: edge_name.to_string(),
                    });
                }
                let producer_vertex = graph.vertex(producer).expect("bound producer exists");
                let consumer_vertex = graph.vertex(consumer).expect("bound consumer exists");
                if let (Some(producer_media), Some(consumer_media)) = (
                    pad_media(producer_vertex, PadDirection::Output, producer_pad)?,
                    pad_media(consumer_vertex, PadDirection::Input, consumer_pad)?,
                ) {
                    check_pad_media(producer_media, consumer_media, producer_pad, consumer_pad)
                        .map_err(CoreError::Invalid)?;
                }
                Some((
                    producer.clone(),
                    producer_pad.clone(),
                    consumer.clone(),
                    consumer_pad.clone(),
                ))
            }
            _ => None,
        };

        match direction {
            PadDirection::Input => record.consumer = Some(endpoint),
            PadDirection::Output => record.producer = Some(endpoint),
        }
        match direction {
            PadDirection::Input => {
                graph
                    .vertex_mut(node_name)
                    .expect("vertex checked above")
                    .sources
                    .insert(pad.to_string(), edge.clone());
            }
            PadDirection::Output => {
                graph
                    .vertex_mut(node_name)
                    .expect("vertex checked above")
                    .sinks
                    .insert(pad.to_string(), edge.clone());
            }
        }

        if let Some((producer, producer_pad, consumer, consumer_pad)) = complete {
            graph
                .add_edge(EdgeLink {
                    name: edge_name.to_string(),
                    producer: producer.clone(),
                    producer_pad,
                    consumer: consumer.clone(),
                    consumer_pad,
                    edge: edge.clone(),
                })
                .map_err(CoreError::Operation)?;
            record.linked = true;
            edge.rearm_spec();
            graph.refresh_direct_tails(&producer);
            graph.refresh_direct_tails(&consumer);
        } else {
            graph.refresh_direct_tails(node_name);
        }
        drop(edges);
        drop(graph);

        match direction {
            PadDirection::Input => {
                node.bind_source(pad, generation_reader(edge.clone(), node_generation))
            }
            PadDirection::Output => {
                let active = edge.writer_generation();
                let binding_generation = active.max(node_generation);
                if active != binding_generation {
                    edge.restart(
                        active,
                        binding_generation,
                        crate::graph::EdgeRestart::Egress,
                    );
                }
                node.bind_sink(pad, generation_writer(edge.clone(), binding_generation));
            }
        }
        Ok(EdgeInstance {
            name: edge_name.to_string(),
            edge,
        })
    }

    pub fn connect_edge(
        &self,
        name: &str,
        producer: &str,
        producer_pad: &str,
        consumer: &str,
        consumer_pad: &str,
        coupling: EdgeKind,
    ) -> Result<EdgeInstance, CoreError> {
        self.ensure_topology_idle(&[producer, consumer])?;
        let (producer_generation, consumer_generation) = {
            let nodes = self.nodes.lock().unwrap();
            let producer_generation = nodes
                .get(producer)
                .ok_or_else(|| CoreError::NotFound {
                    kind: "node",
                    name: producer.to_string(),
                })?
                .active_generation;
            let consumer_generation = nodes
                .get(consumer)
                .ok_or_else(|| CoreError::NotFound {
                    kind: "node",
                    name: consumer.to_string(),
                })?
                .active_generation;
            (producer_generation, consumer_generation)
        };
        let mut graph = self.graph.lock().unwrap();
        if self.edges.lock().unwrap().contains_key(name) || graph.edge(name).is_some() {
            return Err(CoreError::AlreadyExists {
                kind: "edge",
                name: name.to_string(),
            });
        }
        let producer_vertex = graph.vertex(producer).ok_or_else(|| CoreError::NotFound {
            kind: "node",
            name: producer.to_string(),
        })?;
        let consumer_vertex = graph.vertex(consumer).ok_or_else(|| CoreError::NotFound {
            kind: "node",
            name: consumer.to_string(),
        })?;
        if let (Some(producer_media), Some(consumer_media)) = (
            pad_media(producer_vertex, PadDirection::Output, producer_pad)?,
            pad_media(consumer_vertex, PadDirection::Input, consumer_pad)?,
        ) {
            check_pad_media(producer_media, consumer_media, producer_pad, consumer_pad)
                .map_err(CoreError::Invalid)?;
        }
        let producer_node = producer_vertex.node.clone();
        let consumer_node = consumer_vertex.node.clone();
        if matches!(coupling, EdgeKind::Direct)
            && (producer_node.kind() != NodeKind::Poll || consumer_node.kind() != NodeKind::Poll)
        {
            return Err(CoreError::Invalid(
                "DirectEdge requires Poll producer and consumer".into(),
            ));
        }
        let edge: Arc<dyn Edge> = match coupling {
            EdgeKind::Direct => {
                let hop = Arc::new(DirectEdge::new());
                hop.set_consumer(consumer_node.clone());
                hop
            }
            EdgeKind::Buffered { capacity } => Arc::new(BufferedEdge::new(capacity)),
        };

        graph
            .add_edge(EdgeLink {
                name: name.to_string(),
                producer: producer.to_string(),
                producer_pad: producer_pad.to_string(),
                consumer: consumer.to_string(),
                consumer_pad: consumer_pad.to_string(),
                edge: edge.clone(),
            })
            .map_err(CoreError::Operation)?;
        graph
            .vertex_mut(producer)
            .expect("producer checked above")
            .sinks
            .insert(producer_pad.to_string(), edge.clone());
        graph
            .vertex_mut(consumer)
            .expect("consumer checked above")
            .sources
            .insert(consumer_pad.to_string(), edge.clone());
        graph.refresh_direct_tails(producer);
        graph.refresh_direct_tails(consumer);
        self.edges.lock().unwrap().insert(
            name.to_string(),
            EdgeRecord {
                edge: edge.clone(),
                producer: Some((producer.to_string(), producer_pad.to_string())),
                consumer: Some((consumer.to_string(), consumer_pad.to_string())),
                linked: true,
            },
        );
        drop(graph);
        let active = edge.writer_generation();
        let producer_generation = active.max(producer_generation);
        if active != producer_generation {
            edge.restart(
                active,
                producer_generation,
                crate::graph::EdgeRestart::Egress,
            );
        }
        producer_node.bind_sink(
            producer_pad,
            generation_writer(edge.clone(), producer_generation),
        );
        consumer_node.bind_source(
            consumer_pad,
            generation_reader(edge.clone(), consumer_generation),
        );
        edge.rearm_spec();
        Ok(EdgeInstance {
            name: name.to_string(),
            edge,
        })
    }

    pub fn destroy_edge(&self, name: &str) -> Result<(), CoreError> {
        let mut graph = self.graph.lock().unwrap();
        let record =
            self.edges
                .lock()
                .unwrap()
                .remove(name)
                .ok_or_else(|| CoreError::NotFound {
                    kind: "edge",
                    name: name.to_string(),
                })?;
        graph.remove_edge(name);
        graph.unbind_edge(&record.edge);
        Ok(())
    }

    pub fn create_group(&self, name: &str) -> Result<(), CoreError> {
        let mut groups = self.groups.lock().unwrap();
        if groups.contains_key(name) {
            return Err(CoreError::AlreadyExists {
                kind: "group",
                name: name.to_string(),
            });
        }
        let group = Arc::new(Group::new(name.to_string(), self.graph.clone()));
        self.install_group_restart_hook(name, &group);
        groups.insert(name.to_string(), group);
        Ok(())
    }

    fn ensure_topology_idle(&self, node_names: &[&str]) -> Result<(), CoreError> {
        let groups = self
            .groups
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for group in groups {
            if group
                .members()
                .iter()
                .any(|member| node_names.contains(&member.as_str()))
                && (group.state() != crate::GroupState::Idle || group.has_active_reconstruction())
            {
                return Err(CoreError::Invalid(format!(
                    "cannot mutate topology while group `{}` is active",
                    group.name()
                )));
            }
        }
        Ok(())
    }

    fn install_group_restart_hook(&self, group_name: &str, group: &Group) {
        let weak = Arc::downgrade(&self.inner);
        let group_name = group_name.to_string();
        group.set_restart_hook(Arc::new(move |request| {
            let inner = weak
                .upgrade()
                .ok_or_else(|| "instance was destroyed during reconstruction".to_string())?;
            let instance = Instance::from_inner(inner);
            instance
                .reconstruct_group(&group_name)
                .map_err(|error| error.to_string())?;
            Ok(request.reconstructed())
        }));
        let weak = Arc::downgrade(&self.inner);
        group.set_action_hook(Arc::new(move |action| {
            let Some(inner) = weak.upgrade() else {
                return;
            };
            let hook = inner.supervisor_action_hook.lock().unwrap().clone();
            if let Some(hook) = hook {
                hook(action);
                return;
            }
            match action {
                crate::SupervisorAction::Panic { message, .. } => {
                    Instance::from_inner(inner).shutdown();
                    panic!("{message}");
                }
                crate::SupervisorAction::Exit { .. } => {
                    Instance::from_inner(inner).shutdown();
                }
            }
        }));
    }

    pub fn add_group_member(&self, group_name: &str, node_name: &str) -> Result<(), CoreError> {
        let node = self.node(node_name).ok_or_else(|| CoreError::NotFound {
            kind: "node",
            name: node_name.to_string(),
        })?;
        if node.restart != RestartPolicy::Off || node.on_error.is_some() {
            let groups = self.groups.lock().unwrap();
            if let Some(other) = groups.values().find(|candidate| {
                candidate.name() != group_name
                    && candidate.members().iter().any(|member| member == node_name)
            }) {
                return Err(CoreError::Invalid(format!(
                    "policy-bearing node `{node_name}` must belong to exactly one group; already belongs to `{}`",
                    other.name()
                )));
            }
        }
        let group = self
            .groups
            .lock()
            .unwrap()
            .get(group_name)
            .cloned()
            .ok_or_else(|| CoreError::NotFound {
                kind: "group",
                name: group_name.to_string(),
            })?;
        if group.state() != crate::GroupState::Idle || group.has_active_reconstruction() {
            return Err(CoreError::Invalid(format!(
                "cannot add node `{node_name}` to running group `{group_name}`"
            )));
        }
        self.install_group_restart_hook(group_name, &group);
        group.add_full(
            node_name,
            node.exec_ctx,
            node.restart,
            node.on_error,
            node.service_hint,
        );
        Ok(())
    }

    fn validate_group_policy_membership(&self, group: &Group) -> Result<(), CoreError> {
        let members = group.members();
        let nodes = self.nodes.lock().unwrap();
        let policy_members = members
            .iter()
            .filter(|name| {
                nodes.get(*name).is_some_and(|node| {
                    node.restart != RestartPolicy::Off || node.on_error.is_some()
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        drop(nodes);
        let groups = self.groups.lock().unwrap();
        for node_name in policy_members {
            let memberships = groups
                .values()
                .filter(|candidate| {
                    candidate
                        .members()
                        .iter()
                        .any(|member| member == &node_name)
                })
                .count();
            if memberships != 1 {
                return Err(CoreError::Invalid(format!(
                    "policy-bearing node `{node_name}` must belong to exactly one group before start or restart; found {memberships}"
                )));
            }
        }
        Ok(())
    }

    pub fn remove_group_member(&self, group_name: &str, node_name: &str) -> Result<(), CoreError> {
        let group = self
            .groups
            .lock()
            .unwrap()
            .get(group_name)
            .cloned()
            .ok_or_else(|| CoreError::NotFound {
                kind: "group",
                name: group_name.to_string(),
            })?;
        if group.state() != crate::GroupState::Idle || group.has_active_reconstruction() {
            return Err(CoreError::Invalid(format!(
                "cannot remove node `{node_name}` from running group `{group_name}`"
            )));
        }
        group.remove(node_name);
        Ok(())
    }

    pub fn start_group(&self, name: &str) -> Result<(), CoreError> {
        let group = self
            .groups
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| CoreError::NotFound {
                kind: "group",
                name: name.to_string(),
            })?;
        self.validate_group_policy_membership(&group)?;
        if group.generation() != 0 && group.state() == crate::GroupState::Idle {
            self.reconstruct_group(name)?;
        }
        group.start().map_err(CoreError::Operation)
    }

    pub fn stop_group(&self, name: &str) -> Result<(), CoreError> {
        let group = self
            .groups
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| CoreError::NotFound {
                kind: "group",
                name: name.to_string(),
            })?;
        group.stop();
        Ok(())
    }

    pub fn restart_group(&self, name: &str) -> Result<(), CoreError> {
        let group = self.group(name).ok_or_else(|| CoreError::NotFound {
            kind: "group",
            name: name.to_string(),
        })?;
        self.validate_group_policy_membership(&group)?;
        group.restart().map_err(CoreError::Operation)
    }

    pub fn group_status(&self, name: &str) -> Result<crate::GroupStatus, CoreError> {
        self.group(name)
            .map(|group| group.status())
            .ok_or_else(|| CoreError::NotFound {
                kind: "group",
                name: name.to_string(),
            })
    }

    pub fn destroy_group(&self, name: &str) -> Result<(), CoreError> {
        let group =
            self.groups
                .lock()
                .unwrap()
                .remove(name)
                .ok_or_else(|| CoreError::NotFound {
                    kind: "group",
                    name: name.to_string(),
                })?;
        group.stop();
        Ok(())
    }
}
