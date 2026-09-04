// `--features async` for the Poll nodes, and the default build for `Media::Stub`,
// which the crate defines only when libav is compiled out.
#![cfg(all(feature = "async", not(feature = "ffmpeg")))]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use avplumber_f7k::services::clock::SyncGroup;
use avplumber_f7k::services::correction::CorrectionGroup;
use avplumber_f7k::{
    AvpMediaType, AvpRational, BuildCtx, Edge, EdgeEvent, EdgeItem, EdgeKind, GroupState, Instance,
    Media, Node, NodeBody, NodeError, NodeKind, NodePads, NodePhase, NodePollContext, NodeRequest,
    NodeSpec, PadDecl, Push, RestartPolicy, Tick, Ts, register_factory, register_spec,
};

const STEP: i64 = 20;
const TB: AvpRational = AvpRational { num: 1, den: 1000 };

static ACTIVE: OnceLock<Mutex<Option<Weak<HarnessState>>>> = OnceLock::new();

#[derive(Default)]
struct HarnessState {
    factory_calls: AtomicUsize,
    starts: Mutex<Vec<String>>,
    writers: Mutex<Vec<(usize, String, Arc<dyn Edge>)>>,
    clocks: Mutex<Vec<Arc<dyn SyncGroup>>>,
    corrections: Mutex<Vec<Arc<CorrectionGroup>>>,
    output_pts: Mutex<Vec<i64>>,
    fallback_video: AtomicUsize,
    fallback_audio: AtomicUsize,
    crossed_eof: AtomicUsize,
    recovery_factory_entered: AtomicBool,
    recovery_gate: Mutex<bool>,
    recovery_ready: Condvar,
    output_ready: Condvar,
    raw_seen: Mutex<Vec<i64>>,
    raw_ready: Condvar,
    rebases: Mutex<Vec<(i64, i64)>>,
    delay_entered: AtomicBool,
    delay_done: AtomicBool,
}

fn state() -> Arc<HarnessState> {
    ACTIVE
        .get()
        .unwrap()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .upgrade()
        .unwrap()
}

fn pads(inputs: &[(&str, AvpMediaType)], outputs: &[(&str, AvpMediaType)]) -> NodePads {
    NodePads {
        sources: inputs
            .iter()
            .map(|(name, media)| PadDecl {
                name: (*name).into(),
                media: *media,
            })
            .collect(),
        sinks: outputs
            .iter()
            .map(|(name, media)| PadDecl {
                name: (*name).into(),
                media: *media,
            })
            .collect(),
    }
}

fn stub(kind: AvpMediaType, pts: i64) -> Media {
    Media::Stub { kind, pts }
}

#[derive(serde::Deserialize)]
struct RestartableInputSpec {}

struct RestartableInput {
    name: String,
    build: usize,
    video: OnceLock<Arc<dyn Edge>>,
    audio: OnceLock<Arc<dyn Edge>>,
    state: Arc<HarnessState>,
}

impl NodeSpec for RestartableInputSpec {
    const TYPE_NAME: &'static str = "continuity_input";
    type Node = RestartableInput;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let state = state();
        let build = state.factory_calls.fetch_add(1, Ordering::SeqCst) + 1;
        let clock = ctx.clock("continuity");
        let correction = ctx.correction("continuity");
        if build == 1 {
            clock.reset(0, TB);
            correction.set_output_tb(TB);
            correction.set_start_ts(Ts { val: 0, tb: TB });
        }
        if build == 2 || build == 3 {
            return Err(format!("planned reconstruction failure {build}"));
        }
        if build == 4 {
            state
                .recovery_factory_entered
                .store(true, Ordering::Release);
            let mut released = state.recovery_gate.lock().unwrap();
            while !*released {
                released = state.recovery_ready.wait(released).unwrap();
            }
        }
        state.clocks.lock().unwrap().push(clock);
        state.corrections.lock().unwrap().push(correction);
        Ok(RestartableInput {
            name: name.into(),
            build,
            video: OnceLock::new(),
            audio: OnceLock::new(),
            state,
        })
    }
}

impl Node for RestartableInput {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        pads(
            &[],
            &[
                ("video", AvpMediaType::VIDEO),
                ("audio", AvpMediaType::AUDIO),
            ],
        )
    }
    fn bind_sink(&self, name: &str, edge: Arc<dyn Edge>) {
        self.state
            .writers
            .lock()
            .unwrap()
            .push((self.build, name.to_string(), edge.clone()));
        match name {
            "video" => {
                let _ = self.video.set(edge);
            }
            "audio" => {
                let _ = self.audio.set(edge);
            }
            _ => unreachable!(),
        }
    }
    fn start(&self) {
        self.state.starts.lock().unwrap().push(self.name.clone());
    }
    fn take_body(self: Arc<Self>) -> NodeBody {
        let mut produced = 0_i64;
        let mut deadline = Instant::now();
        NodeBody::Poll(Box::new(move |ctx| {
            if self.build == 1 && produced == 4 {
                return Err(NodeError::new(
                    self.name.clone(),
                    NodePhase::Process,
                    "forced input failure",
                ));
            }
            let now = Instant::now();
            if now < deadline {
                ctx.wait_deadline(deadline);
                return Ok(Tick::Idle);
            }
            let origin = self.build as i64 * 10_000;
            let pts = origin + produced * STEP;
            let video = self.video.get().unwrap();
            let audio = self.audio.get().unwrap();
            if video.push(stub(AvpMediaType::VIDEO, pts)) == Push::Closed
                || audio.push(stub(AvpMediaType::AUDIO, pts)) == Push::Closed
            {
                return Ok(Tick::Done);
            }
            produced += 1;
            deadline += Duration::from_millis(STEP as u64);
            Ok(Tick::Again)
        }))
    }
}

struct Sentinel {
    name: String,
    video_input: OnceLock<Arc<dyn Edge>>,
    audio_input: OnceLock<Arc<dyn Edge>>,
    video_output: OnceLock<Arc<dyn Edge>>,
    audio_output: OnceLock<Arc<dyn Edge>>,
    correction: Arc<CorrectionGroup>,
    clock: Arc<dyn SyncGroup>,
    state: Arc<HarnessState>,
    pending_raw: Mutex<(Option<i64>, Option<i64>)>,
}

impl Node for Sentinel {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        pads(
            &[
                ("video_in", AvpMediaType::VIDEO),
                ("audio_in", AvpMediaType::AUDIO),
            ],
            &[
                ("video_out", AvpMediaType::VIDEO),
                ("audio_out", AvpMediaType::AUDIO),
            ],
        )
    }
    fn bind_source(&self, name: &str, edge: Arc<dyn Edge>) {
        if name == "video_in" {
            let _ = self.video_input.set(edge);
        } else {
            let _ = self.audio_input.set(edge);
        }
    }
    fn bind_sink(&self, name: &str, edge: Arc<dyn Edge>) {
        if name == "video_out" {
            let _ = self.video_output.set(edge);
        } else {
            let _ = self.audio_output.set(edge);
        }
    }
    fn start(&self) {
        self.state.starts.lock().unwrap().push(self.name.clone());
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let video_input = self.video_input.get().unwrap();
        let audio_input = self.audio_input.get().unwrap();
        for (input, video) in [(video_input, true), (audio_input, false)] {
            while let Some(item) = input.try_take() {
                match item {
                    EdgeItem::Buffer(media) => {
                        let raw = media.ts().val;
                        self.state.raw_seen.lock().unwrap().push(raw);
                        self.state.raw_ready.notify_all();
                        let mut pending = self.pending_raw.lock().unwrap();
                        if video {
                            pending.0 = Some(raw);
                        } else {
                            pending.1 = Some(raw);
                        }
                    }
                    EdgeItem::Event(EdgeEvent::Eof) => {
                        self.state.crossed_eof.fetch_add(1, Ordering::SeqCst);
                    }
                    EdgeItem::Event(_) => {}
                }
            }
        }

        self.correction.advance_live(self.clock.as_ref());
        let cursor = self.correction.member_cursor(&self.name).unwrap();
        let expected = self.correction.snapshot().expected_ts;
        let pts = if cursor.next_ts.is_valid() {
            cursor.next_ts.val
        } else {
            expected.val.div_euclid(STEP) * STEP
        };
        if pts > expected.val {
            let cursor_wall = self.clock.map_to_wall(pts, TB);
            let expected_wall = self.clock.map_to_wall(expected.val, expected.tb);
            let delay_us = cursor_wall.saturating_sub(expected_wall).max(1) as u64;
            ctx.wait_readable(video_input.clone());
            ctx.wait_readable(audio_input.clone());
            ctx.wait_deadline(Instant::now() + Duration::from_micros(delay_us));
            return Tick::Idle;
        }
        let video_output = self.video_output.get().unwrap();
        let audio_output = self.audio_output.get().unwrap();
        if video_output.is_full() || audio_output.is_full() {
            ctx.wait_writable(video_output.clone());
            ctx.wait_writable(audio_output.clone());
            return Tick::Idle;
        }
        if video_output.push(stub(AvpMediaType::VIDEO, pts)) != Push::Accepted
            || audio_output.push(stub(AvpMediaType::AUDIO, pts)) != Push::Accepted
        {
            return Tick::Done;
        }

        let pending = std::mem::take(&mut *self.pending_raw.lock().unwrap());
        match pending {
            (Some(video), Some(audio)) => {
                self.state
                    .rebases
                    .lock()
                    .unwrap()
                    .push((video.min(audio), pts));
            }
            (video, audio) => {
                if video.is_none() {
                    self.state.fallback_video.fetch_add(1, Ordering::SeqCst);
                }
                if audio.is_none() {
                    self.state.fallback_audio.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
        self.correction
            .commit(
                &self.name,
                Ts {
                    val: pts + STEP,
                    tb: TB,
                },
                cursor.generation,
            )
            .expect("Sentinel owns the only correction member");
        Tick::Again
    }
}

struct Forward {
    name: String,
    kind: AvpMediaType,
    input: OnceLock<Arc<dyn Edge>>,
    output: OnceLock<Arc<dyn Edge>>,
    state: Arc<HarnessState>,
}

impl Node for Forward {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        pads(&[("in", self.kind)], &[("out", self.kind)])
    }
    fn bind_source(&self, _name: &str, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn bind_sink(&self, _name: &str, edge: Arc<dyn Edge>) {
        let _ = self.output.set(edge);
    }
    fn start(&self) {
        self.state.starts.lock().unwrap().push(self.name.clone());
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let input = self.input.get().unwrap();
        match input.try_take() {
            Some(EdgeItem::Buffer(media)) => match self.output.get().unwrap().push(media) {
                Push::Accepted => Tick::Again,
                Push::Full => {
                    ctx.wait_writable(self.output.get().unwrap().clone());
                    Tick::Idle
                }
                Push::Closed | Push::Dropped => Tick::Done,
            },
            Some(_) => Tick::Again,
            None => {
                ctx.wait_readable(input.clone());
                Tick::Idle
            }
        }
    }
}

struct Mux {
    video: OnceLock<Arc<dyn Edge>>,
    audio: OnceLock<Arc<dyn Edge>>,
    output: OnceLock<Arc<dyn Edge>>,
    pending_video: Mutex<Option<i64>>,
    pending_audio: Mutex<Option<i64>>,
    state: Arc<HarnessState>,
}

impl Node for Mux {
    fn name(&self) -> &str {
        "mux"
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        pads(
            &[
                ("video", AvpMediaType::VIDEO),
                ("audio", AvpMediaType::AUDIO),
            ],
            &[("out", AvpMediaType::PACKET)],
        )
    }
    fn bind_source(&self, name: &str, edge: Arc<dyn Edge>) {
        if name == "video" {
            let _ = self.video.set(edge);
        } else {
            let _ = self.audio.set(edge);
        }
    }
    fn bind_sink(&self, _name: &str, edge: Arc<dyn Edge>) {
        let _ = self.output.set(edge);
    }
    fn start(&self) {
        self.state.starts.lock().unwrap().push("mux".into());
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        if self.pending_video.lock().unwrap().is_none()
            && let Some(EdgeItem::Buffer(media)) = self.video.get().unwrap().try_take()
        {
            *self.pending_video.lock().unwrap() = Some(media.ts().val);
        }
        if self.pending_audio.lock().unwrap().is_none()
            && let Some(EdgeItem::Buffer(media)) = self.audio.get().unwrap().try_take()
        {
            *self.pending_audio.lock().unwrap() = Some(media.ts().val);
        }
        let video = *self.pending_video.lock().unwrap();
        let audio = *self.pending_audio.lock().unwrap();
        if let (Some(video), Some(audio)) = (video, audio) {
            assert_eq!(video, audio);
            match self
                .output
                .get()
                .unwrap()
                .push(stub(AvpMediaType::PACKET, video))
            {
                Push::Accepted => {
                    *self.pending_video.lock().unwrap() = None;
                    *self.pending_audio.lock().unwrap() = None;
                    return Tick::Again;
                }
                Push::Closed | Push::Dropped => return Tick::Done,
                Push::Full => {
                    ctx.wait_writable(self.output.get().unwrap().clone());
                    return Tick::Idle;
                }
            }
        }
        ctx.wait_readable(self.video.get().unwrap().clone());
        ctx.wait_readable(self.audio.get().unwrap().clone());
        Tick::Idle
    }
}

struct Output {
    input: OnceLock<Arc<dyn Edge>>,
    state: Arc<HarnessState>,
}

impl Node for Output {
    fn name(&self) -> &str {
        "output"
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn pads(&self) -> NodePads {
        pads(&[("in", AvpMediaType::PACKET)], &[])
    }
    fn bind_source(&self, _name: &str, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn start(&self) {
        self.state.starts.lock().unwrap().push("output".into());
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        let input = self.input.get().unwrap();
        match input.try_take() {
            Some(EdgeItem::Buffer(media)) => {
                self.state.output_pts.lock().unwrap().push(media.ts().val);
                self.state.output_ready.notify_all();
                Tick::Again
            }
            Some(_) => Tick::Again,
            None => {
                ctx.wait_readable(input.clone());
                Tick::Idle
            }
        }
    }
}

struct DelayNode {
    trigger: Arc<dyn Edge>,
    state: Arc<HarnessState>,
}

impl Node for DelayNode {
    fn name(&self) -> &str {
        "delay"
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        if self.trigger.try_take().is_none() {
            ctx.wait_readable(self.trigger.clone());
            return Tick::Idle;
        }
        self.state.delay_entered.store(true, Ordering::Release);
        std::thread::sleep(Duration::from_millis(STEP as u64 * 5));
        self.state.delay_done.store(true, Ordering::Release);
        Tick::Done
    }
}

fn wait_for(timeout: Duration, what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while !ready() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn wait_for_output(state: &HarnessState, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut output = state.output_pts.lock().unwrap();
    while output.len() < count {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "timed out waiting for output {count}");
        let (next, timeout) = state.output_ready.wait_timeout(output, remaining).unwrap();
        output = next;
        assert!(!timeout.timed_out() || output.len() >= count);
    }
}

fn wait_for_raw(state: &HarnessState, raw: i64) {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut seen = state.raw_seen.lock().unwrap();
    while !seen.contains(&raw) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "timed out waiting for raw PTS {raw}");
        let (next, timeout) = state.raw_ready.wait_timeout(seen, remaining).unwrap();
        seen = next;
        assert!(!timeout.timed_out() || seen.contains(&raw));
    }
}

fn add_node(inst: &Instance, ty: &str, name: &str) {
    inst.create_node(NodeRequest::new(ty, name, serde_json::json!({})))
        .unwrap();
}

#[test]
fn output_continues_across_failed_retries_recovery_and_manual_restart() {
    let state = Arc::new(HarnessState::default());
    *ACTIVE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(Arc::downgrade(&state));
    let inst = Instance::new();
    register_spec::<RestartableInputSpec>(&inst);
    let mut input = NodeRequest::new(
        RestartableInputSpec::TYPE_NAME,
        "input",
        serde_json::json!({}),
    );
    input.restart = Some(RestartPolicy::RestartGroup);
    inst.create_node(input).unwrap();

    let (clock, correction) = {
        let clocks = state.clocks.lock().unwrap();
        let corrections = state.corrections.lock().unwrap();
        (clocks[0].clone(), corrections[0].clone())
    };
    correction.register("sentinel");
    let initial_cursor = correction.member_cursor("sentinel").unwrap();
    correction
        .commit("sentinel", Ts { val: 0, tb: TB }, initial_cursor.generation)
        .unwrap();
    {
        let state = state.clone();
        let clock = clock.clone();
        let correction = correction.clone();
        register_factory(&inst, "sentinel", move |name, _| {
            Ok(Arc::new(Sentinel {
                name: name.into(),
                video_input: OnceLock::new(),
                audio_input: OnceLock::new(),
                video_output: OnceLock::new(),
                audio_output: OnceLock::new(),
                correction: correction.clone(),
                clock: clock.clone(),
                state: state.clone(),
                pending_raw: Mutex::new((None, None)),
            }))
        });
    }
    register_factory(&inst, "forward", {
        let state = state.clone();
        move |name, _| {
            Ok(Arc::new(Forward {
                name: name.into(),
                kind: AvpMediaType::VIDEO,
                input: OnceLock::new(),
                output: OnceLock::new(),
                state: state.clone(),
            }))
        }
    });
    register_factory(&inst, "mux", {
        let state = state.clone();
        move |_, _| {
            Ok(Arc::new(Mux {
                video: OnceLock::new(),
                audio: OnceLock::new(),
                output: OnceLock::new(),
                pending_video: Mutex::new(None),
                pending_audio: Mutex::new(None),
                state: state.clone(),
            }))
        }
    });
    register_factory(&inst, "output", {
        let state = state.clone();
        move |_, _| {
            Ok(Arc::new(Output {
                input: OnceLock::new(),
                state: state.clone(),
            }))
        }
    });
    let delay_trigger: Arc<dyn Edge> = Arc::new(avplumber_f7k::BufferedEdge::new(1));
    register_factory(&inst, "delay", {
        let state = state.clone();
        let trigger = delay_trigger.clone();
        move |_, _| {
            Ok(Arc::new(DelayNode {
                trigger: trigger.clone(),
                state: state.clone(),
            }))
        }
    });
    for (ty, name) in [
        ("sentinel", "sentinel"),
        ("forward", "encoder"),
        ("mux", "mux"),
        ("output", "output"),
        ("delay", "delay"),
    ] {
        add_node(&inst, ty, name);
    }
    let video_boundary = inst
        .connect_edge(
            "input-video",
            "input",
            "video",
            "sentinel",
            "video_in",
            EdgeKind::Buffered { capacity: 8 },
        )
        .unwrap()
        .edge;
    let audio_boundary = inst
        .connect_edge(
            "input-audio",
            "input",
            "audio",
            "sentinel",
            "audio_in",
            EdgeKind::Buffered { capacity: 8 },
        )
        .unwrap()
        .edge;
    for (name, producer, producer_pad, consumer, consumer_pad) in [
        ("corrected-video", "sentinel", "video_out", "encoder", "in"),
        ("encoded-video", "encoder", "out", "mux", "video"),
        ("corrected-audio", "sentinel", "audio_out", "mux", "audio"),
        ("muxed", "mux", "out", "output", "in"),
    ] {
        inst.connect_edge(
            name,
            producer,
            producer_pad,
            consumer,
            consumer_pad,
            EdgeKind::Buffered { capacity: 4 },
        )
        .unwrap();
    }
    inst.create_group("persistent").unwrap();
    for name in ["sentinel", "encoder", "mux", "output", "delay"] {
        inst.add_group_member("persistent", name).unwrap();
    }
    inst.create_group("input").unwrap();
    inst.add_group_member("input", "input").unwrap();

    inst.start_group("persistent").unwrap();
    inst.start_group("input").unwrap();
    let input_group = inst.group("input").unwrap();
    wait_for_output(&state, 5);
    let before_delay = state.output_pts.lock().unwrap().len();
    delay_trigger.push_event(EdgeEvent::FlushStop);
    wait_for(Duration::from_secs(1), "delayed poll to enter", || {
        state.delay_entered.load(Ordering::Acquire)
    });
    wait_for(Duration::from_secs(1), "delayed poll to finish", || {
        state.delay_done.load(Ordering::Acquire)
    });
    wait_for_output(&state, before_delay + 8);
    wait_for(
        Duration::from_secs(5),
        "recovery factory gate after two failed retries",
        || state.recovery_factory_entered.load(Ordering::Acquire),
    );
    assert_eq!(state.factory_calls.load(Ordering::SeqCst), 4);
    assert_eq!(input_group.generation(), 1);
    let during_outage = state.output_pts.lock().unwrap().len();
    wait_for_output(&state, during_outage + 10);
    let outage_last = *state.output_pts.lock().unwrap().last().unwrap();
    assert_ne!(input_group.state(), GroupState::Running);
    {
        *state.recovery_gate.lock().unwrap() = true;
        state.recovery_ready.notify_all();
    }
    wait_for(Duration::from_secs(4), "two retries and recovery", || {
        state.factory_calls.load(Ordering::SeqCst) >= 4
            && input_group.generation() == 2
            && input_group.state() == GroupState::Running
    });
    wait_for(Duration::from_secs(2), "recovered input rebase", || {
        state
            .rebases
            .lock()
            .unwrap()
            .iter()
            .any(|(raw, _)| *raw >= 40_000)
    });
    let first_recovered = state
        .rebases
        .lock()
        .unwrap()
        .iter()
        .find(|(raw, _)| *raw >= 40_000)
        .copied()
        .unwrap();
    assert!(first_recovered.1 > outage_last);
    assert!(first_recovered.0 >= 40_000);
    assert_eq!(input_group.status().restart_count, 1);
    assert_eq!(state.factory_calls.load(Ordering::SeqCst), 4);
    assert!(state.fallback_video.load(Ordering::SeqCst) >= 10);
    assert!(state.fallback_audio.load(Ordering::SeqCst) >= 10);
    assert_eq!(state.crossed_eof.load(Ordering::SeqCst), 0);
    assert!(Arc::ptr_eq(
        &video_boundary,
        &inst.edge_link("input-video").unwrap().edge
    ));
    assert!(Arc::ptr_eq(
        &audio_boundary,
        &inst.edge_link("input-audio").unwrap().edge
    ));
    let writers = state.writers.lock().unwrap().clone();
    let stale_video = writers
        .iter()
        .find(|(build, pad, _)| *build == 1 && pad == "video")
        .unwrap()
        .2
        .clone();
    let stale_audio = writers
        .iter()
        .find(|(build, pad, _)| *build == 1 && pad == "audio")
        .unwrap()
        .2
        .clone();
    assert_eq!(
        stale_video.push(stub(AvpMediaType::VIDEO, -1)),
        Push::Closed
    );
    assert_eq!(
        stale_audio.push(stub(AvpMediaType::AUDIO, -1)),
        Push::Closed
    );
    stale_video.push_event(EdgeEvent::Eof);
    stale_audio.push_event(EdgeEvent::Eof);
    let active_video = writers
        .iter()
        .find(|(build, pad, _)| *build == 4 && pad == "video")
        .unwrap()
        .2
        .clone();
    assert_eq!(
        active_video.push(stub(AvpMediaType::VIDEO, 49_999)),
        Push::Accepted
    );
    wait_for_raw(&state, 49_999);
    assert_eq!(
        state.crossed_eof.load(Ordering::SeqCst),
        0,
        "stale generation EOF crossed the restartable boundary"
    );

    let before_manual = state.output_pts.lock().unwrap().len();
    inst.restart_group("input").unwrap();
    wait_for(Duration::from_secs(2), "manual restart", || {
        input_group.generation() == 3 && input_group.state() == GroupState::Running
    });
    wait_for(
        Duration::from_secs(1),
        "output after manual restart",
        || state.output_pts.lock().unwrap().len() >= before_manual + 5,
    );
    wait_for(Duration::from_secs(2), "manual generation input", || {
        state
            .raw_seen
            .lock()
            .unwrap()
            .iter()
            .any(|pts| *pts >= 50_000)
    });
    assert_eq!(state.factory_calls.load(Ordering::SeqCst), 5);
    assert_eq!(input_group.status().restart_count, 2);
    assert_eq!(state.crossed_eof.load(Ordering::SeqCst), 0);

    let starts = state.starts.lock().unwrap();
    for persistent in ["sentinel", "encoder", "mux", "output"] {
        assert_eq!(
            starts
                .iter()
                .filter(|name| name.as_str() == persistent)
                .count(),
            1,
            "{persistent} restarted"
        );
    }
    drop(starts);
    let clocks = state.clocks.lock().unwrap();
    let corrections = state.corrections.lock().unwrap();
    assert_eq!(clocks.len(), 3);
    assert_eq!(corrections.len(), 3);
    assert!(
        clocks
            .iter()
            .all(|candidate| Arc::ptr_eq(candidate, &clock))
    );
    assert!(
        corrections
            .iter()
            .all(|candidate| Arc::ptr_eq(candidate, &correction))
    );
    drop(clocks);
    drop(corrections);
    assert!(correction.snapshot().expected_ts.val > outage_last);

    let pts = state.output_pts.lock().unwrap().clone();
    assert!(pts.len() > 100);
    for pair in pts.windows(2) {
        assert_eq!(pair[1], pair[0] + STEP);
    }
    for raw_origin in [10_000, 40_000, 50_000] {
        assert!(
            !pts.contains(&raw_origin),
            "raw generation origin {raw_origin} leaked to output"
        );
    }

    let stopping = Instant::now();
    inst.stop_group("input").unwrap();
    inst.stop_group("persistent").unwrap();
    assert!(stopping.elapsed() < Duration::from_secs(1));
    *ACTIVE.get().unwrap().lock().unwrap() = None;
}
