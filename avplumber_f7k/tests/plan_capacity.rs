//! `queue.plan_capacity`: by name, and `*` as the default for every edge not
//! planned by name — through both ways an edge comes to exist, an explicit
//! lookup and a node's `src`/`dst` binding.

#![cfg(not(feature = "ffmpeg"))]

use std::sync::Arc;

use avplumber_f7k::graph::edge::Push;
use avplumber_f7k::graph::{AvpMediaType, Media, Node};
use avplumber_f7k::{Instance, control, register_factory};

struct Stub {
    name: String,
}

impl Node for Stub {
    fn name(&self) -> &str {
        &self.name
    }
}

fn stub(pts: i64) -> Media {
    Media::Stub {
        kind: AvpMediaType::VIDEO,
        pts,
    }
}

/// How many buffers the edge takes before answering `Full`.
fn room(inst: &Instance, edge: &str) -> usize {
    let edge = inst.named_edge(edge);
    let mut n = 0;
    while edge.offer(stub(n as i64)).is_ok() {
        n += 1;
        assert!(n < 1000, "the edge never fills");
    }
    assert!(matches!(edge.offer(stub(-1)), Err((Push::Full, _))));
    n
}

#[test]
fn the_wildcard_is_the_default_and_a_name_overrides_it() {
    let inst = Instance::new();
    register_factory(&inst, "stub", |name, _| {
        Ok(Arc::new(Stub { name: name.into() }) as Arc<dyn Node>)
    });
    control::exec_line(&inst, "queue.plan_capacity * 1").unwrap();
    control::exec_line(&inst, "queue.plan_capacity wide 3").unwrap();

    // Edges created by binding a node's `dst` / `src`.
    control::exec_line(
        &inst,
        r#"node.add {"type":"stub","name":"a","dst":"narrow"}"#,
    )
    .unwrap();
    control::exec_line(
        &inst,
        r#"node.add {"type":"stub","name":"b","src":"narrow","dst":"wide"}"#,
    )
    .unwrap();
    control::exec_line(&inst, r#"node.add {"type":"stub","name":"c","src":"wide"}"#).unwrap();
    assert_eq!(
        room(&inst, "narrow"),
        1,
        "`*` applies to an edge bound by name"
    );
    assert_eq!(room(&inst, "wide"), 3, "a planned name wins over `*`");

    // And by a direct lookup.
    assert_eq!(room(&inst, "other"), 1);
}

#[test]
fn without_a_plan_the_edge_has_its_own_default() {
    let inst = Instance::new();
    assert_eq!(room(&inst, "e"), 64);
}
