//! Topological sort over a group's nodes. Start order is sources first, then
//! downstream; stop order is the reverse. Returns Err on a cycle.

use std::collections::{HashMap, HashSet};

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::graph::EdgeLink;

/// Topo-sort the given node names using the edge links (producer -> consumer).
/// Returns the start order (producer before consumer). Returns Err(message)
/// if a cycle is found among the given nodes.
pub fn topo_sort(nodes: &[String], links: &[EdgeLink]) -> Result<Vec<String>, String> {
    let node_set: HashSet<&String> = nodes.iter().collect();
    // NodeIndex is assigned sequentially in slice order, so idx.index() maps
    // back to the name by position.
    let idx_of: HashMap<&String, NodeIndex> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n, NodeIndex::new(i)))
        .collect();
    let mut graph: DiGraph<(), ()> = DiGraph::with_capacity(nodes.len(), links.len());
    for _ in 0..nodes.len() {
        graph.add_node(());
    }
    for l in links {
        if node_set.contains(&l.producer) && node_set.contains(&l.consumer) {
            if let (Some(&p), Some(&c)) = (idx_of.get(&l.producer), idx_of.get(&l.consumer)) {
                graph.add_edge(p, c, ());
            }
        }
    }
    match toposort(&graph, None) {
        Ok(order) => Ok(order
            .into_iter()
            .map(|idx| nodes[idx.index()].clone())
            .collect()),
        Err(cycle) => {
            let idx = cycle.node_id();
            Err(format!("cycle detected at node {}", nodes[idx.index()]))
        }
    }
}
