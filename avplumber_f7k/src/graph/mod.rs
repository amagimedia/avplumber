//! Graph substrate: topology records and runtime edge handles. Does not run
//! nodes; the supervisor does.

pub mod buffer;
pub mod buffered_edge;
pub mod capability;
pub mod direct_edge;
pub mod edge;
pub mod error;
pub mod media;
pub mod node;
pub mod pad;
pub mod poll_ctx;
pub mod routing;
pub mod spec;
pub mod timebase;

pub use buffer::{AVP_NOPTS, AvpMediaType, AvpMediaVtable, AvpRational};
pub use buffered_edge::BufferedEdge;
pub use capability::{AvpInterfaceId, AvpServiceId};
pub use direct_edge::DirectEdge;
pub use edge::{
    Edge, EdgeEvent, EdgeHint, EdgeHintCell, EdgeItem, EdgeKind, EdgeRestart, EdgeWaker, Push,
    Wakeup, generation_reader, generation_writer,
};
pub use error::{NodeError, NodePhase};
pub use media::{Media, OpaqueFrame, Ts};
pub use node::{Blocked, Node, NodeBody, NodeFuture, NodeKind, Tick};
pub use pad::{In, NodePads, Out, PadDecl, check_pad_media};
pub use poll_ctx::NodePollContext;
pub use spec::{CatalogStream, ChannelLayout, MuxStream, PacketSpec, Spec, StreamSelection};

use std::collections::HashMap;
use std::sync::Arc;

pub struct Vertex {
    pub name: String,
    pub node: Arc<dyn Node>,
    pub sources: HashMap<String, Arc<dyn Edge>>,
    pub sinks: HashMap<String, Arc<dyn Edge>>,
    pub source_media: HashMap<String, AvpMediaType>,
    pub sink_media: HashMap<String, AvpMediaType>,
}

pub struct Pad {
    pub name: String,
    pub media: AvpMediaType,
    pub capacity: usize,
}

#[derive(Clone)]
pub struct EdgeLink {
    pub name: String,
    pub producer: String,
    pub producer_pad: String,
    pub consumer: String,
    pub consumer_pad: String,
    pub edge: Arc<dyn Edge>,
}

pub struct Graph {
    vertices: HashMap<String, Vertex>,
    links: Vec<EdgeLink>,
    named_edges: HashMap<String, Arc<dyn Edge>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            vertices: HashMap::new(),
            links: Vec::new(),
            named_edges: HashMap::new(),
        }
    }

    pub fn add_vertex(&mut self, v: Vertex) -> Result<(), String> {
        if self.vertices.contains_key(&v.name) {
            return Err(format!("node name busy: {}", v.name));
        }
        self.vertices.insert(v.name.clone(), v);
        Ok(())
    }

    pub fn add_edge(&mut self, link: EdgeLink) -> Result<(), String> {
        if self
            .links
            .iter()
            .any(|l| l.name == link.name && !link.name.is_empty())
        {
            return Err(format!("edge name busy: {}", link.name));
        }
        if !link.name.is_empty() {
            self.named_edges
                .insert(link.name.clone(), link.edge.clone());
        }
        self.links.push(link);
        Ok(())
    }

    pub fn remove_vertex(&mut self, name: &str) {
        self.vertices.remove(name);
        let mut removed = Vec::new();
        self.links.retain(|l| {
            let drop = l.producer == name || l.consumer == name;
            if drop && !l.name.is_empty() {
                self.named_edges.remove(&l.name);
            }
            if drop {
                removed.push(l.edge.clone());
            }
            !drop
        });
        self.unbind_edges(&removed);
    }

    pub fn remove_edge(&mut self, name: &str) {
        let mut removed = self
            .named_edges
            .remove(name)
            .into_iter()
            .collect::<Vec<_>>();
        self.links.retain(|l| {
            if l.name == name {
                removed.push(l.edge.clone());
                false
            } else {
                true
            }
        });
        self.unbind_edges(&removed);
    }

    pub(crate) fn unbind_edge(&mut self, edge: &Arc<dyn Edge>) {
        self.unbind_edges(std::slice::from_ref(edge));
    }

    fn unbind_edges(&mut self, removed: &[Arc<dyn Edge>]) {
        for v in self.vertices.values_mut() {
            v.sources
                .retain(|_, edge| !removed.iter().any(|old| Arc::ptr_eq(old, edge)));
            v.sinks
                .retain(|_, edge| !removed.iter().any(|old| Arc::ptr_eq(old, edge)));
        }
        let names: Vec<String> = self.vertices.keys().cloned().collect();
        for name in names {
            self.refresh_direct_tails(&name);
        }
    }

    /// Direct inputs of `name` wait on that node's unique output, if any.
    pub(crate) fn refresh_direct_tails(&self, name: &str) {
        let Some(v) = self.vertices.get(name) else {
            return;
        };
        let outs: Vec<Arc<dyn Edge>> = v.sinks.values().cloned().collect();
        let tail = (outs.len() == 1).then(|| outs[0].clone());
        for src in v.sources.values() {
            src.attach_direct_tail(tail.clone());
        }
    }

    pub(crate) fn refresh_direct_bindings(&self) {
        for link in &self.links {
            if !link.edge.is_direct() {
                continue;
            }
            if let Some(consumer) = self.vertices.get(&link.consumer) {
                link.edge.attach_direct_consumer(consumer.node.clone());
            }
        }
        for name in self.vertices.keys() {
            self.refresh_direct_tails(name);
        }
    }

    pub fn vertex(&self, name: &str) -> Option<&Vertex> {
        self.vertices.get(name)
    }
    pub fn vertex_mut(&mut self, name: &str) -> Option<&mut Vertex> {
        self.vertices.get_mut(name)
    }
    pub fn edge(&self, name: &str) -> Option<&Arc<dyn Edge>> {
        self.named_edges
            .get(name)
            .or_else(|| self.links.iter().find(|l| l.name == name).map(|l| &l.edge))
    }
    pub fn links(&self) -> &[EdgeLink] {
        &self.links
    }
    pub fn vertices(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.values()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
