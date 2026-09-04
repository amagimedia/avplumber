//! The consumer → producer hint back-channel: per-variant latching, only
//! changes reported, and state that outlives a restart. Default build.

#![cfg(not(feature = "ffmpeg"))]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

use avplumber_f7k::{
    AvpMediaType, BufferedEdge, DirectEdge, Edge, EdgeHint, EdgeRestart, EdgeWaker, Media, Push,
    StreamSelection, generation_reader, generation_writer,
};

fn stub(pts: i64) -> Media {
    Media::Stub {
        kind: AvpMediaType::VIDEO,
        pts,
    }
}

fn edges() -> Vec<Arc<dyn Edge>> {
    vec![
        Arc::new(BufferedEdge::new(2)) as Arc<dyn Edge>,
        Arc::new(DirectEdge::new()) as Arc<dyn Edge>,
    ]
}

fn selection(enabled: &[i32]) -> StreamSelection {
    StreamSelection::new(enabled.to_vec())
}

#[test]
fn hints_latch_per_variant_and_only_changes_are_reported() {
    for edge in edges() {
        edge.post_hint(EdgeHint::StreamsFilter("a:0".into()));
        assert_eq!(
            edge.take_hints(),
            vec![EdgeHint::StreamsFilter("a:0".into())]
        );
        assert!(
            edge.take_hints().is_empty(),
            "a drained hint must not be reported again"
        );

        edge.post_hint(EdgeHint::Streams(selection(&[0, 2])));
        assert_eq!(
            edge.take_hints(),
            vec![EdgeHint::Streams(selection(&[0, 2]))],
            "a Streams hint must not re-report the StreamsFilter one"
        );

        // Last write wins within a variant, without disturbing the other.
        edge.post_hint(EdgeHint::StreamsFilter("v".into()));
        edge.post_hint(EdgeHint::StreamsFilter("v:1".into()));
        assert_eq!(
            edge.take_hints(),
            vec![EdgeHint::StreamsFilter("v:1".into())]
        );
    }
}

#[test]
fn a_hint_posted_before_the_producer_asks_is_still_delivered() {
    for edge in edges() {
        // demux posts from `bind_source`, i.e. at script-parse time, long
        // before the producing node's first iteration.
        edge.post_hint(EdgeHint::StreamsFilter("a".into()));
        edge.post_hint(EdgeHint::Streams(selection(&[1])));

        let hints = edge.take_hints();
        assert_eq!(hints.len(), 2, "{hints:?}");
        assert!(hints.contains(&EdgeHint::StreamsFilter("a".into())));
        assert!(hints.contains(&EdgeHint::Streams(selection(&[1]))));
    }
}

#[test]
fn hints_cross_the_generation_wrappers_unfenced() {
    let direct: Arc<dyn Edge> = Arc::new(DirectEdge::new());
    let consumer = generation_reader(direct.clone(), 1);
    let producer = generation_writer(direct.clone(), 1);
    consumer.post_hint(EdgeHint::Streams(selection(&[3])));
    assert_eq!(
        producer.take_hints(),
        vec![EdgeHint::Streams(selection(&[3]))]
    );

    // A hint is current state, so a wrapper for another generation still sees
    // it: the rebuilt endpoint is exactly the one that needs to learn it.
    consumer.post_hint(EdgeHint::StreamsFilter("d".into()));
    assert_eq!(
        generation_writer(direct.clone(), 2).take_hints(),
        vec![EdgeHint::StreamsFilter("d".into())]
    );
}

#[test]
fn a_restart_re_arms_hints_for_the_new_producer() {
    for kind in [
        EdgeRestart::Internal,
        EdgeRestart::Egress,
        EdgeRestart::Ingress,
    ] {
        for edge in edges() {
            edge.post_hint(EdgeHint::StreamsFilter("a:0".into()));
            edge.post_hint(EdgeHint::Streams(selection(&[0, 1])));
            assert_eq!(edge.take_hints().len(), 2);

            edge.restart(1, 2, kind);

            let hints = edge.take_hints();
            assert_eq!(
                hints.len(),
                2,
                "{kind:?}: a fresh producer has never drained the cell"
            );
            assert!(hints.contains(&EdgeHint::StreamsFilter("a:0".into())));
            assert!(hints.contains(&EdgeHint::Streams(selection(&[0, 1]))));
        }
    }
}

struct FlagWaker(Arc<AtomicBool>);

impl EdgeWaker for FlagWaker {
    fn wake(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
fn posting_a_hint_wakes_the_producer_side() {
    for edge in edges() {
        let woken = Arc::new(AtomicBool::new(false));
        edge.notify_writable(Box::new(FlagWaker(woken.clone())));
        edge.post_hint(EdgeHint::StreamsFilter("v:0".into()));
        assert!(
            woken.load(Ordering::SeqCst),
            "a producer parked on a full edge must still answer a hint"
        );
    }
}

#[test]
fn a_pending_hint_is_writable_readiness_so_a_full_edge_cannot_park_the_answer() {
    // Poll and async producers park until the edge reports writable. If a full
    // edge stayed "not ready", the producer would never run to answer the hint
    // the consumer is waiting on — while the consumer waits to free space.
    for edge in edges() {
        assert!(!edge.has_hints());
        edge.post_hint(EdgeHint::StreamsFilter("v:0".into()));
        assert!(edge.has_hints(), "a posted hint makes the edge writable");
        assert!(!edge.take_hints().is_empty());
        assert!(!edge.has_hints(), "a drained hint is no longer readiness");
    }
}

struct CondvarWaker {
    signal: Arc<(Mutex<bool>, Condvar)>,
}

impl EdgeWaker for CondvarWaker {
    fn wake(&self) {
        *self.signal.0.lock().unwrap() = true;
        self.signal.1.notify_all();
    }
}

#[test]
fn a_hint_reaches_a_producer_parked_on_a_full_edge() {
    // The one interleaving that could deadlock: the producer is out of space
    // while the consumer waits for an answer only the producer can give.
    let edge: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
    assert_eq!(edge.push(stub(1)), Push::Accepted);
    assert!(edge.is_full());

    let seen: Arc<Mutex<Vec<EdgeHint>>> = Arc::new(Mutex::new(Vec::new()));
    let signal = Arc::new((Mutex::new(false), Condvar::new()));
    let (parked_tx, parked_rx) = mpsc::channel();
    let producer = {
        let edge = edge.clone();
        let seen = seen.clone();
        let signal = signal.clone();
        std::thread::spawn(move || {
            // A blocking producer's own writable park: install the waker, then
            // wait on it, then drain hints before retrying the push.
            for _ in 0..100 {
                edge.notify_writable(Box::new(CondvarWaker {
                    signal: signal.clone(),
                }));
                let hints = edge.take_hints();
                if !hints.is_empty() {
                    seen.lock().unwrap().extend(hints);
                    return;
                }
                let mut woken = signal.0.lock().unwrap();
                parked_tx.send(()).unwrap();
                if !*woken {
                    let (guard, _) = signal
                        .1
                        .wait_timeout(woken, Duration::from_millis(50))
                        .unwrap();
                    woken = guard;
                }
                *woken = false;
            }
        })
    };
    parked_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    edge.post_hint(EdgeHint::Streams(selection(&[2])));
    producer.join().unwrap();

    assert_eq!(
        seen.lock().unwrap().as_slice(),
        [EdgeHint::Streams(selection(&[2]))],
        "the answer must not wait for the edge to drain"
    );
    assert!(edge.is_full(), "and no space was freed to get it");
}
