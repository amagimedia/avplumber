//! Shift consensus for a correction group.
//!
//! The step decides which input moment occupies the next ideal-grid slot. It
//! does not build frames. Video spends a residual as a whole-frame drop or
//! repeat. Audio spends the same residual as a sample stretch. A jump past
//! `rebase_threshold` stamps the real input on the cursor and rewrites that
//! member's shift in one step.

use std::collections::BTreeMap;

/// How member shifts move toward the group shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Convergence {
    /// Group shift steps toward the median. A hard rebase moves that median by
    /// at most one slew step.
    Slew,
    /// A hard rebase sets the group shift to the member that jumped. Everyone
    /// else spends budget to follow. The median does not pull the group back.
    SlewFollow,
    /// Group shift copies the anchor member. Others follow it.
    Anchor,
    /// Copy the group onto the member, then copy the member back onto the group
    /// when that misses the cursor. One propose finishes the correction.
    Snap,
    /// A forward hole repeats up to five quanta, then rebases. An overlap drops
    /// the input and leaves the cursor.
    DropFill,
}

/// Which quantum a member can spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    /// Drop or repeat one whole frame.
    Video,
    /// Stretch by the slew step. A one-quantum drop or repeat is not the slight
    /// path.
    Audio,
}

#[derive(Clone, Debug)]
pub struct Policy {
    pub mode: Convergence,
    /// Seconds of shift corrected per second of media.
    pub max_slew: f64,
    /// Ticks. A larger continuity break rebases.
    pub rebase_threshold: i64,
    /// Ticks. Inside this, the shift is left alone.
    pub deadband: i64,
    pub anchor: Option<String>,
    pub locked: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            mode: Convergence::Slew,
            max_slew: 0.05,
            rebase_threshold: 500,
            deadband: 1,
            anchor: None,
            locked: false,
        }
    }
}

#[derive(Clone, Debug)]
struct Member {
    kind: MemberKind,
    /// Ticks of one drop or repeat. `0` means "use the proposal duration".
    quantum: i64,
    local_shift: Option<i64>,
    /// Fractional slew budget. 1000 units are one tick.
    residual_milli: i64,
    /// Previous consumed input PTS, for the continuity check.
    last_input: Option<i64>,
    last_duration: i64,
    /// Backup frames already emitted for the current `DropFill` hole.
    fills: u32,
}

#[derive(Clone, Debug)]
pub struct ConvergeState {
    policy: Policy,
    configured: bool,
    group_shift: Option<i64>,
    members: BTreeMap<String, Member>,
}

/// One real input, already in the group's timebase.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub member: String,
    pub input_pts: i64,
    pub next_ts: i64,
    pub duration: i64,
    /// Media time since this member's previous arrival. `0` on the follow-up
    /// after a repeat, so a single arrival spends at most one quantum.
    pub dt: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Stamp the input on the cursor.
    Emit { emit_pts: i64 },
    /// Discard the input. The cursor stays.
    Drop,
    /// Emit one repeated quantum at the cursor. The input stays pending.
    Repeat { emit_pts: i64 },
    /// Stamp the input on the cursor and time-scale it by `sample_delta` ticks.
    Stretch { emit_pts: i64, sample_delta: i64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub action: Action,
    pub local_shift: i64,
    pub group_shift: i64,
    /// Cursor after this action.
    pub next_ts: i64,
    /// The input jumped past `rebase_threshold`. The audio executor rebuilds
    /// the resampler instead of stretching across that jump.
    pub rebased: bool,
}

impl ConvergeState {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            configured: true,
            group_shift: None,
            members: BTreeMap::new(),
        }
    }

    /// No mode yet. The first [`Self::configure`] sets it; a later member that
    /// names a different mode is rejected.
    pub fn unconfigured() -> Self {
        Self {
            policy: Policy::default(),
            configured: false,
            group_shift: None,
            members: BTreeMap::new(),
        }
    }

    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// First configuration wins. A later call with a different mode is an error
    /// so two members of one group cannot correct by different rules.
    pub fn configure(&mut self, policy: Policy) -> Result<(), String> {
        if self.configured && policy.mode != self.policy.mode {
            return Err(format!(
                "correction group mode {:?} disagrees with {:?}",
                policy.mode, self.policy.mode
            ));
        }
        if !self.configured {
            self.policy = policy;
            self.configured = true;
        } else {
            // Same mode: a second member may tighten the shared deadband by
            // leaving the already chosen policy in place.
            self.policy.locked = self.policy.locked || policy.locked;
            if self.policy.anchor.is_none() {
                self.policy.anchor = policy.anchor;
            }
        }
        Ok(())
    }

    pub fn lock(&mut self) {
        self.policy.locked = true;
    }

    /// Replace the slew limits without changing the mode. The group calls this
    /// when its timebase changes, so a threshold given in seconds stays the
    /// same length of time.
    pub fn set_rates(&mut self, max_slew: f64, rebase_threshold: i64, deadband: i64) {
        self.policy.max_slew = max_slew;
        self.policy.rebase_threshold = rebase_threshold.max(1);
        self.policy.deadband = deadband.max(1);
    }

    pub fn group_shift(&self) -> Option<i64> {
        self.group_shift
    }

    pub fn local_shift(&self, member: &str) -> Option<i64> {
        self.members.get(member).and_then(|m| m.local_shift)
    }

    pub fn register(&mut self, name: &str, kind: MemberKind, quantum: i64) {
        self.members.entry(name.to_string()).or_insert(Member {
            kind,
            quantum,
            local_shift: None,
            residual_milli: 0,
            last_input: None,
            last_duration: 0,
            fills: 0,
        });
    }

    pub fn unregister(&mut self, name: &str) {
        self.members.remove(name);
    }

    pub fn step(&mut self, proposal: &Proposal) -> Result<Decision, String> {
        if proposal.duration <= 0 {
            return Err("correction duration must be positive".into());
        }
        let kind = self
            .members
            .get(&proposal.member)
            .map(|m| m.kind)
            .ok_or_else(|| format!("unknown member {}", proposal.member))?;
        let quantum = self
            .members
            .get(&proposal.member)
            .map(|m| m.quantum)
            .unwrap_or(0);
        let quantum = if quantum > 0 {
            quantum
        } else {
            proposal.duration
        };

        if self.group_shift.is_none() || self.members[&proposal.member].local_shift.is_none() {
            return Ok(self.seed(proposal));
        }

        // `obs` is the shift this input would stamp if emitted on the cursor:
        // cursor − input. A lasting offset keeps showing up here after the
        // packet that introduced it, so the budget is spent until `obs` meets
        // the group shift.
        let obs = proposal.next_ts - proposal.input_pts;
        let local = self.members[&proposal.member].local_shift.unwrap();

        if self.policy.mode == Convergence::Snap {
            return Ok(self.snap(proposal, obs));
        }
        if self.policy.mode == Convergence::DropFill {
            return Ok(self.drop_fill(proposal, obs, local, quantum));
        }
        if (obs - local).abs() > self.policy.rebase_threshold {
            return Ok(self.rebase(proposal, obs));
        }

        let group = self.group_shift.unwrap();
        let err = group - obs;
        if err.abs() <= self.policy.deadband {
            let member = self.members.get_mut(&proposal.member).unwrap();
            member.local_shift = Some(obs);
            member.residual_milli = 0;
            member.fills = 0;
            member.last_input = Some(proposal.input_pts);
            member.last_duration = proposal.duration;
            let next_ts = proposal.next_ts + proposal.duration;
            self.nudge_group(&proposal.member, false, proposal.dt);
            return Ok(self.decision(
                Action::Emit {
                    emit_pts: proposal.next_ts,
                },
                next_ts,
            ));
        }

        let budget = slew_budget_milli(self.policy.max_slew, proposal.dt);
        let add_milli = clamp(err.saturating_mul(1000), -budget, budget);
        let member = self.members.get_mut(&proposal.member).unwrap();
        if proposal.dt > 0 {
            member.residual_milli += add_milli;
        }
        let quantum_milli = quantum.saturating_mul(1000);

        let act = if proposal.dt > 0 && kind == MemberKind::Audio {
            let add = member.residual_milli / 1000;
            if add != 0 {
                member.residual_milli -= add * 1000;
                // This frame's stamp is still `obs`. The extra cursor movement
                // changes the next stamp by `add`.
                member.local_shift = Some(obs);
                member.last_input = Some(proposal.input_pts);
                member.last_duration = proposal.duration;
                Action::Stretch {
                    emit_pts: proposal.next_ts,
                    sample_delta: add,
                }
            } else {
                member.local_shift = Some(obs);
                member.last_input = Some(proposal.input_pts);
                member.last_duration = proposal.duration;
                Action::Emit {
                    emit_pts: proposal.next_ts,
                }
            }
        } else if proposal.dt > 0
            && kind == MemberKind::Video
            && member.residual_milli >= quantum_milli
            && err > 0
        {
            member.residual_milli -= quantum_milli;
            Action::Repeat {
                emit_pts: proposal.next_ts,
            }
        } else if proposal.dt > 0
            && kind == MemberKind::Video
            && member.residual_milli <= -quantum_milli
            && err < 0
        {
            member.residual_milli += quantum_milli;
            member.last_input = Some(proposal.input_pts);
            member.last_duration = proposal.duration;
            Action::Drop
        } else {
            member.local_shift = Some(obs);
            member.last_input = Some(proposal.input_pts);
            member.last_duration = proposal.duration;
            Action::Emit {
                emit_pts: proposal.next_ts,
            }
        };

        let next_ts = match &act {
            Action::Drop => proposal.next_ts,
            Action::Repeat { .. } => proposal.next_ts + quantum,
            Action::Stretch { sample_delta, .. } => {
                proposal.next_ts + proposal.duration + sample_delta
            }
            Action::Emit { .. } => proposal.next_ts + proposal.duration,
        };
        self.nudge_group(&proposal.member, false, proposal.dt);
        Ok(self.decision(act, next_ts))
    }

    /// Stall fill. Advances the cursor with backup slots and does not touch the
    /// shift: a missing input has no PTS to form one.
    pub fn fill(
        &mut self,
        member: &str,
        cursor: i64,
        expected: i64,
        timeout: i64,
        quantum: i64,
        max_frames: u32,
    ) -> Result<Vec<i64>, String> {
        if !self.members.contains_key(member) {
            return Err(format!("unknown member {member}"));
        }
        if quantum <= 0 {
            return Err("fill quantum must be positive".into());
        }
        if expected - cursor <= timeout {
            return Ok(Vec::new());
        }
        let mut pts = Vec::new();
        let mut at = cursor;
        while at < expected && pts.len() < max_frames as usize {
            pts.push(at);
            at += quantum;
        }
        Ok(pts)
    }

    fn seed(&mut self, proposal: &Proposal) -> Decision {
        let obs = proposal.next_ts - proposal.input_pts;
        if self.group_shift.is_none() {
            self.group_shift = Some(obs);
        }
        let member = self.members.get_mut(&proposal.member).unwrap();
        member.local_shift = Some(obs);
        member.last_input = Some(proposal.input_pts);
        member.last_duration = proposal.duration;
        member.residual_milli = 0;
        let next_ts = proposal.next_ts + proposal.duration;
        // The first observation of a member is adopted whole. Startup does not
        // slew in from zero. The group still takes at most one step toward the
        // new median when another member is already there.
        let seeded_group = self.group_shift == Some(obs);
        if !seeded_group {
            self.nudge_group(&proposal.member, false, proposal.dt);
        }
        self.decision(
            Action::Emit {
                emit_pts: proposal.next_ts,
            },
            next_ts,
        )
    }

    fn rebase(&mut self, proposal: &Proposal, obs: i64) -> Decision {
        let member = self.members.get_mut(&proposal.member).unwrap();
        member.local_shift = Some(obs);
        member.residual_milli = 0;
        member.fills = 0;
        member.last_input = Some(proposal.input_pts);
        member.last_duration = proposal.duration;
        let next_ts = proposal.next_ts + proposal.duration;
        self.nudge_group(&proposal.member, true, proposal.dt);
        let mut decision = self.decision(
            Action::Emit {
                emit_pts: proposal.next_ts,
            },
            next_ts,
        );
        decision.rebased = true;
        decision
    }

    fn snap(&mut self, proposal: &Proposal, obs: i64) -> Decision {
        let group = self.group_shift.unwrap();
        let mut local = self.members[&proposal.member].local_shift.unwrap();
        if (group - local).abs() > self.policy.deadband {
            local = group;
        }
        if (obs - local).abs() > self.policy.deadband {
            local = obs;
            if !self.policy.locked {
                self.group_shift = Some(local);
            }
        }
        let member = self.members.get_mut(&proposal.member).unwrap();
        member.local_shift = Some(local);
        member.residual_milli = 0;
        member.last_input = Some(proposal.input_pts);
        member.last_duration = proposal.duration;
        let next_ts = proposal.next_ts + proposal.duration;
        self.decision(
            Action::Emit {
                emit_pts: proposal.next_ts,
            },
            next_ts,
        )
    }

    fn drop_fill(&mut self, proposal: &Proposal, obs: i64, local: i64, quantum: i64) -> Decision {
        let jump = obs - local;
        if jump > self.policy.deadband {
            let member = self.members.get_mut(&proposal.member).unwrap();
            member.fills = 0;
            member.last_input = Some(proposal.input_pts);
            member.last_duration = proposal.duration;
            return self.decision(Action::Drop, proposal.next_ts);
        }
        if jump < -self.policy.deadband {
            let gap = -jump;
            let member = self.members.get_mut(&proposal.member).unwrap();
            if member.fills < 5 && gap > quantum {
                member.fills += 1;
                member.local_shift = Some(local + quantum);
                return self.decision(
                    Action::Repeat {
                        emit_pts: proposal.next_ts,
                    },
                    proposal.next_ts + quantum,
                );
            }
            return self.rebase(proposal, obs);
        }
        let member = self.members.get_mut(&proposal.member).unwrap();
        member.fills = 0;
        member.last_input = Some(proposal.input_pts);
        member.last_duration = proposal.duration;
        self.decision(
            Action::Emit {
                emit_pts: proposal.next_ts,
            },
            proposal.next_ts + proposal.duration,
        )
    }

    fn nudge_group(&mut self, member: &str, rebased: bool, dt: i64) {
        if self.policy.locked {
            return;
        }
        let local = match self.members.get(member).and_then(|m| m.local_shift) {
            Some(local) => local,
            None => return,
        };
        let group = match self.group_shift {
            Some(group) => group,
            None => return,
        };
        match self.policy.mode {
            // The member that jumped becomes the target. Later proposes do not
            // drag the group back toward the median.
            Convergence::SlewFollow if rebased => {
                self.group_shift = Some(local);
            }
            Convergence::Anchor => {
                if self.policy.anchor.as_deref() == Some(member) {
                    self.group_shift = Some(local);
                }
            }
            Convergence::Slew => {
                let step = slew_step(self.policy.max_slew, dt).max(0);
                let locals: Vec<i64> = self
                    .members
                    .iter()
                    .filter_map(|(name, m)| {
                        let shift = m.local_shift?;
                        Some(if rebased && name == member {
                            clamp(shift, group - step, group + step)
                        } else {
                            shift
                        })
                    })
                    .collect();
                if locals.is_empty() || step == 0 {
                    return;
                }
                let med = median(&locals);
                self.group_shift = Some(group + clamp(med - group, -step, step));
            }
            Convergence::Snap | Convergence::DropFill | Convergence::SlewFollow => {}
        }
    }

    fn decision(&self, action: Action, next_ts: i64) -> Decision {
        Decision {
            action,
            local_shift: 0,
            group_shift: self.group_shift.unwrap_or(0),
            next_ts,
            rebased: false,
        }
    }
}

fn slew_step(max_slew: f64, dt: i64) -> i64 {
    slew_budget_milli(max_slew, dt) / 1000
}

/// Slew budget in thousandths of a tick, so a step smaller than one tick still
/// accumulates.
fn slew_budget_milli(max_slew: f64, dt: i64) -> i64 {
    if dt <= 0 || max_slew <= 0.0 {
        return 0;
    }
    (max_slew * dt as f64 * 1000.0).round() as i64
}

fn clamp(value: i64, lo: i64, hi: i64) -> i64 {
    value.max(lo).min(hi)
}

fn median(vals: &[i64]) -> i64 {
    let mut vals = vals.to_vec();
    vals.sort_unstable();
    let n = vals.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2
    }
}

/// The decision's `local_shift` is the member that just stepped. `decision()`
/// cannot see that name, so step() overwrites it. This helper keeps the
/// overwrite next to the return.
fn finish(state: &ConvergeState, member: &str, mut decision: Decision) -> Decision {
    decision.local_shift = state.local_shift(member).unwrap_or(0);
    decision.group_shift = state.group_shift().unwrap_or(0);
    decision
}

// The methods above return `self.decision(...)` with a zero local shift.
// Patching every return through `finish` is done by wrapping `step`.
impl ConvergeState {
    pub fn propose(&mut self, proposal: &Proposal) -> Result<Decision, String> {
        let member = proposal.member.clone();
        let decision = self.step(proposal)?;
        Ok(finish(self, &member, decision))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(mode: Convergence) -> Policy {
        Policy {
            mode,
            max_slew: 0.05,
            rebase_threshold: 500,
            deadband: 1,
            anchor: None,
            locked: false,
        }
    }

    fn state(mode: Convergence) -> ConvergeState {
        let mut state = ConvergeState::new(policy(mode));
        state.register("video", MemberKind::Video, 40);
        state.register("audio", MemberKind::Audio, 1);
        state
    }

    fn proposal(member: &str, input: i64, next: i64, dur: i64, dt: i64) -> Proposal {
        Proposal {
            member: member.into(),
            input_pts: input,
            next_ts: next,
            duration: dur,
            dt,
        }
    }

    fn emit_pts(action: &Action) -> Option<i64> {
        match action {
            Action::Emit { emit_pts } | Action::Repeat { emit_pts } | Action::Stretch { emit_pts, .. } => {
                Some(*emit_pts)
            }
            Action::Drop => None,
        }
    }

    #[test]
    fn continuous_video_stays_on_the_nominal_grid() {
        let mut state = state(Convergence::Slew);
        let mut cursor = 0;
        let mut prev = None;
        for i in 0..25 {
            let input = i * 40;
            let decision = state
                .propose(&proposal("video", input, cursor, 40, 40))
                .unwrap();
            let pts = emit_pts(&decision.action).unwrap();
            assert_eq!(pts, cursor, "emit leaves the cursor");
            if let Some(prev) = prev {
                assert_eq!(pts - prev, 40);
            }
            prev = Some(pts);
            cursor = decision.next_ts;
        }
    }

    #[test]
    fn fifteen_ms_split_reaches_the_deadband_within_the_slew_budget() {
        let mut state = ConvergeState::new(policy(Convergence::Slew));
        state.register("a", MemberKind::Video, 5);
        state.register("b", MemberKind::Video, 5);
        // Seed A at shift 0 and B at shift 15. First observations are adopted
        // whole, which is what opens the split.
        state.propose(&proposal("a", 0, 0, 5, 5)).unwrap();
        state.propose(&proposal("b", -15, 0, 5, 5)).unwrap();
        assert_eq!(state.local_shift("a"), Some(0));
        assert_eq!(state.local_shift("b"), Some(15));

        let mut corrections = Vec::new();
        let mut cursor_a = 5;
        let mut cursor_b = 5;
        let mut input_a = 5;
        let mut input_b = -10;
        for n in 0..80 {
            let media_now = (n + 1) * 5;
            for (name, cursor, input) in [
                ("a", &mut cursor_a, &mut input_a),
                ("b", &mut cursor_b, &mut input_b),
            ] {
                let mut dt = 5;
                loop {
                    let before = state.local_shift(name);
                    let decision = state
                        .propose(&proposal(name, *input, *cursor, 5, dt))
                        .unwrap();
                    if emit_pts(&decision.action).is_some() {
                        if let Some(pts) = emit_pts(&decision.action) {
                            assert_eq!(pts, *cursor);
                        }
                    }
                    let after = state.local_shift(name);
                    if before != after {
                        if let (Some(prev), Some(media)) = (corrections.last().copied(), Some(media_now))
                        {
                            let min_gap = 5.0 / 0.05;
                            assert!(
                                (media - prev) as f64 + 5.0 >= min_gap - 5.0,
                                "correction at {media} follows {prev}, closer than one quantum/max_slew"
                            );
                        }
                        corrections.push(media_now);
                    }
                    *cursor = decision.next_ts;
                    match decision.action {
                        Action::Repeat { .. } => dt = 0,
                        Action::Drop | Action::Emit { .. } | Action::Stretch { .. } => {
                            *input += 5;
                            break;
                        }
                    }
                }
            }
        }
        let a = state.local_shift("a").unwrap();
        let b = state.local_shift("b").unwrap();
        assert!(
            (a - b).abs() <= 1,
            "shifts {a} and {b} did not meet inside the deadband"
        );
    }

    #[test]
    fn sub_frame_video_error_does_not_drop() {
        let mut state = ConvergeState::new(policy(Convergence::Slew));
        state.register("a", MemberKind::Video, 40);
        state.register("b", MemberKind::Video, 40);
        state.propose(&proposal("a", 0, 0, 40, 40)).unwrap();
        state.propose(&proposal("b", -15, 0, 40, 40)).unwrap();
        let mut drops = 0;
        let mut cursor_b = 40;
        let mut input_b = 25;
        for _ in 0..10 {
            let decision = state
                .propose(&proposal("b", input_b, cursor_b, 40, 40))
                .unwrap();
            if matches!(decision.action, Action::Drop | Action::Repeat { .. }) {
                drops += 1;
            }
            cursor_b = decision.next_ts;
            input_b += 40;
        }
        assert_eq!(drops, 0, "a 15 ms error is smaller than one 40 ms frame");
    }

    #[test]
    fn two_second_jump_rebases_only_that_member() {
        let mut state = state(Convergence::Slew);
        let mut cursor_v = 0;
        let mut cursor_a = 0;
        for i in 0..10 {
            let d = state
                .propose(&proposal("video", i * 40, cursor_v, 40, 40))
                .unwrap();
            cursor_v = d.next_ts;
            let d = state
                .propose(&proposal("audio", i * 20, cursor_a, 20, 20))
                .unwrap();
            cursor_a = d.next_ts;
        }
        let audio_before = state.local_shift("audio").unwrap();
        let group_before = state.group_shift().unwrap();
        // Video input skips 2000 ms. The cursor stays where the grid left it.
        let jumped = cursor_v / 40 * 40 + 2000;
        let decision = state
            .propose(&proposal("video", jumped, cursor_v, 40, 40))
            .unwrap();
        assert_eq!(emit_pts(&decision.action), Some(cursor_v));
        assert_eq!(decision.next_ts, cursor_v + 40);
        assert_ne!(state.local_shift("video").unwrap(), audio_before);
        assert_eq!(state.local_shift("audio").unwrap(), audio_before);
        let group_after = state.group_shift().unwrap();
        let step = slew_step(0.05, 40);
        assert!(
            (group_after - group_before).abs() <= step.max(1),
            "slew moved the group by {} on a rebase",
            group_after - group_before
        );
    }

    #[test]
    fn slew_follow_sets_the_group_and_the_other_member_spends_budget() {
        let mut state = state(Convergence::SlewFollow);
        let mut cursor_v = 0;
        let mut cursor_a = 0;
        for i in 0..5 {
            cursor_v = state
                .propose(&proposal("video", i * 40, cursor_v, 40, 40))
                .unwrap()
                .next_ts;
            cursor_a = state
                .propose(&proposal("audio", i * 20, cursor_a, 20, 20))
                .unwrap()
                .next_ts;
        }
        let jumped = 5 * 40 + 2000;
        let decision = state
            .propose(&proposal("video", jumped, cursor_v, 40, 40))
            .unwrap();
        assert_eq!(emit_pts(&decision.action), Some(cursor_v));
        let video_shift = state.local_shift("video").unwrap();
        assert_eq!(state.group_shift().unwrap(), video_shift);
        let mut input = cursor_a;
        let mut cursor = cursor_a;
        let mut moved = 0;
        for _ in 0..40 {
            let d = state
                .propose(&proposal("audio", input, cursor, 20, 20))
                .unwrap();
            if let Action::Stretch { sample_delta, .. } = d.action {
                assert!(sample_delta.abs() <= slew_step(0.05, 20).max(1));
                assert_ne!(sample_delta, 0);
                moved += 1;
            }
            cursor = d.next_ts;
            input += 20;
        }
        assert!(moved > 0, "audio did not spend budget toward the video shift");
        let obs = cursor - input;
        assert!(
            obs.abs() < video_shift.abs(),
            "audio stamp {obs} did not move toward video shift {video_shift}"
        );
        assert_ne!(obs, 0);
    }

    #[test]
    fn snap_corrects_in_one_propose() {
        let mut state = state(Convergence::Snap);
        let mut cursor = 0;
        for i in 0..3 {
            cursor = state
                .propose(&proposal("video", i * 40, cursor, 40, 40))
                .unwrap()
                .next_ts;
        }
        let jumped = cursor + 2000;
        let decision = state
            .propose(&proposal("video", jumped, cursor, 40, 40))
            .unwrap();
        assert_eq!(emit_pts(&decision.action), Some(cursor));
        let obs = cursor - jumped;
        assert_eq!(decision.local_shift, obs);
        assert_eq!(decision.group_shift, obs);
    }

    #[test]
    fn lock_keeps_the_group_shift() {
        let mut state = state(Convergence::SlewFollow);
        let mut cursor = 0;
        for i in 0..3 {
            cursor = state
                .propose(&proposal("video", i * 40, cursor, 40, 40))
                .unwrap()
                .next_ts;
        }
        let seeded = state.group_shift().unwrap();
        state.lock();
        let jumped = cursor + 2000;
        let decision = state
            .propose(&proposal("video", jumped, cursor, 40, 40))
            .unwrap();
        assert_eq!(emit_pts(&decision.action), Some(cursor));
        assert_eq!(state.group_shift().unwrap(), seeded);
        assert_ne!(decision.local_shift, seeded);
    }

    #[test]
    fn anchor_tracks_only_the_named_member() {
        let mut policy = policy(Convergence::Anchor);
        policy.anchor = Some("video".into());
        let mut state = ConvergeState::new(policy);
        state.register("video", MemberKind::Video, 40);
        state.register("audio", MemberKind::Audio, 1);
        state.propose(&proposal("video", 0, 0, 40, 40)).unwrap();
        state.propose(&proposal("audio", -30, 0, 20, 20)).unwrap();
        assert_eq!(state.group_shift().unwrap(), state.local_shift("video").unwrap());
        let before = state.group_shift().unwrap();
        state.propose(&proposal("audio", -10, 20, 20, 20)).unwrap();
        assert_eq!(state.group_shift().unwrap(), before);
    }

    #[test]
    fn drop_fill_overlap_drops_and_a_large_hole_rebases_after_five() {
        let mut state = ConvergeState::new(policy(Convergence::DropFill));
        state.register("video", MemberKind::Video, 40);
        state.propose(&proposal("video", 0, 0, 40, 40)).unwrap();
        // Input behind the cursor: overlap.
        let behind = state.propose(&proposal("video", 0, 40, 40, 40)).unwrap();
        assert!(matches!(behind.action, Action::Drop));
        assert_eq!(behind.next_ts, 40);

        // Fresh stream, input far ahead: five repeats, then the real frame.
        let mut state = ConvergeState::new(policy(Convergence::DropFill));
        state.register("video", MemberKind::Video, 40);
        state.propose(&proposal("video", 0, 0, 40, 40)).unwrap();
        let mut cursor = 40;
        let mut repeats = 0;
        let ahead = 40 + 1000;
        loop {
            let d = state
                .propose(&proposal("video", ahead, cursor, 40, 40))
                .unwrap();
            cursor = d.next_ts;
            match d.action {
                Action::Repeat { emit_pts } => {
                    assert_eq!(emit_pts + 40, cursor);
                    repeats += 1;
                }
                Action::Emit { emit_pts } => {
                    assert_eq!(emit_pts, cursor - 40);
                    break;
                }
                other => panic!("unexpected {other:?}"),
            }
            assert!(repeats <= 5);
        }
        assert_eq!(repeats, 5);
    }

    #[test]
    fn fill_does_not_move_the_shift() {
        let mut state = state(Convergence::Slew);
        state.propose(&proposal("video", 0, 0, 40, 40)).unwrap();
        let shift = state.group_shift().unwrap();
        let slots = state.fill("video", 40, 40 + 500, 100, 40, 5).unwrap();
        assert_eq!(slots.len(), 5);
        assert_eq!(slots[0], 40);
        assert_eq!(state.group_shift().unwrap(), shift);
        let quiet = state.fill("video", 40, 80, 100, 40, 5).unwrap();
        assert!(quiet.is_empty());
    }

    struct Out {
        content: i64,
        input_pts: i64,
        output_pts: i64,
        duration: i64,
    }

    struct FixtureItem {
        content: i64,
        pts: i64,
    }

    fn broken_items(audio: bool) -> Vec<FixtureItem> {
        let step = if audio { 20 } else { 40 };
        let mut t = 0;
        let mut items = Vec::new();
        while t < 7000 {
            let pts = if audio && (1000..2500).contains(&t) {
                t + 200
            } else if !audio && (2000..2500).contains(&t) {
                t + 1500
            } else {
                t
            };
            items.push(FixtureItem { content: t, pts });
            t += step;
        }
        items
    }

    struct Played {
        video: Vec<Out>,
        audio: Vec<Out>,
        /// Next video and audio cursors, in milliseconds.
        video_cursor: i64,
        audio_cursor: i64,
        state: ConvergeState,
    }

    fn play(mode: Convergence) -> Played {
        let mut policy = policy(mode);
        policy.max_slew = 0.1;
        let mut state = ConvergeState::new(policy);
        state.register("video", MemberKind::Video, 40);
        state.register("audio", MemberKind::Audio, 1);
        let video = broken_items(false);
        let audio = broken_items(true);
        let mut vi = 0;
        let mut ai = 0;
        let mut vc = 0;
        let mut ac = 0;
        let mut v_out = Vec::new();
        let mut a_out = Vec::new();
        let mut guard = 0;
        while vi < video.len() || ai < audio.len() {
            guard += 1;
            assert!(guard < 20_000, "driver did not finish");
            let v_pts = video.get(vi).map(|i| i.pts);
            let a_pts = audio.get(ai).map(|i| i.pts);
            let video_turn = match (v_pts, a_pts) {
                (Some(v), Some(a)) => v <= a,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if video_turn {
                let item = &video[vi];
                let mut dt = 40;
                loop {
                    let d = state
                        .propose(&proposal("video", item.pts, vc, 40, dt))
                        .unwrap();
                    match d.action {
                        Action::Repeat { emit_pts } => {
                            let content = v_out.last().map(|o: &Out| o.content).unwrap_or(item.content);
                            v_out.push(Out {
                                content,
                                input_pts: item.pts,
                                output_pts: emit_pts,
                                duration: 40,
                            });
                            vc = d.next_ts;
                            dt = 0;
                        }
                        Action::Emit { emit_pts } => {
                            v_out.push(Out {
                                content: item.content,
                                input_pts: item.pts,
                                output_pts: emit_pts,
                                duration: d.next_ts - emit_pts,
                            });
                            vc = d.next_ts;
                            vi += 1;
                            break;
                        }
                        Action::Drop => {
                            vi += 1;
                            break;
                        }
                        Action::Stretch { .. } => panic!("video does not stretch"),
                    }
                }
            } else {
                let item = &audio[ai];
                let d = state
                    .propose(&proposal("audio", item.pts, ac, 20, 20))
                    .unwrap();
                match d.action {
                    Action::Stretch { emit_pts, sample_delta } => {
                        let delta = sample_delta;
                        a_out.push(Out {
                            content: item.content,
                            input_pts: item.pts,
                            output_pts: emit_pts,
                            duration: 20 + delta,
                        });
                        ac = d.next_ts;
                        ai += 1;
                    }
                    Action::Emit { emit_pts } => {
                        a_out.push(Out {
                            content: item.content,
                            input_pts: item.pts,
                            output_pts: emit_pts,
                            duration: 20,
                        });
                        ac = d.next_ts;
                        ai += 1;
                    }
                    Action::Drop => ai += 1,
                    Action::Repeat { .. } => panic!("audio stretch path does not repeat"),
                }
            }
        }
        Played {
            video: v_out,
            audio: a_out,
            video_cursor: vc,
            audio_cursor: ac,
            state,
        }
    }

    /// After the scripted run the input stops. A lead inside `timeout` emits
    /// nothing. A lead past it emits five backup slots on the cursor and leaves
    /// the shift alone.
    fn assert_timeout_fill(played: &mut Played) {
        let shift = played.state.group_shift();
        let video_local = played.state.local_shift("video");
        let audio_local = played.state.local_shift("audio");
        let video_cursor = played.video_cursor;
        let audio_cursor = played.audio_cursor;
        let quiet = played
            .state
            .fill("video", video_cursor, video_cursor + 100, 100, 40, 5)
            .unwrap();
        assert!(
            quiet.is_empty(),
            "backup slots while the live clock is still inside timeout"
        );
        let video = played
            .state
            .fill("video", video_cursor, video_cursor + 200, 100, 40, 5)
            .unwrap();
        assert_eq!(
            video,
            (0..5).map(|i| video_cursor + i * 40).collect::<Vec<_>>()
        );
        let audio = played
            .state
            .fill("audio", audio_cursor, audio_cursor + 100, 40, 20, 5)
            .unwrap();
        assert_eq!(
            audio,
            (0..5).map(|i| audio_cursor + i * 20).collect::<Vec<_>>()
        );
        assert_eq!(played.state.group_shift(), shift);
        assert_eq!(played.state.local_shift("video"), video_local);
        assert_eq!(played.state.local_shift("audio"), audio_local);
    }

    fn aligned_contents(from: i64, step: i64, end: i64) -> Vec<i64> {
        let mut t = from;
        if t % step != 0 {
            t += step - (t % step);
        }
        let mut out = Vec::new();
        while t < end {
            out.push(t);
            t += step;
        }
        out
    }

    fn assert_monotonic_grid(outs: &[Out]) {
        for pair in outs.windows(2) {
            assert_eq!(
                pair[1].output_pts - pair[0].output_pts,
                pair[0].duration,
                "grid step {} then {}",
                pair[0].output_pts,
                pair[1].output_pts
            );
        }
    }

    fn assert_tail(played: &Played, tail_from: i64) {
        let v: Vec<&Out> = played
            .video
            .iter()
            .filter(|o| o.content >= tail_from)
            .collect();
        let a: Vec<&Out> = played
            .audio
            .iter()
            .filter(|o| o.content >= tail_from)
            .collect();
        assert!(!v.is_empty() && !a.is_empty());
        for pair in v.windows(2) {
            assert_eq!(pair[1].output_pts - pair[0].output_pts, 40);
        }
        for pair in a.windows(2) {
            assert_eq!(pair[1].output_pts - pair[0].output_pts, 20);
        }
        let v_contents: Vec<i64> = v.iter().map(|o| o.content).collect();
        let a_contents: Vec<i64> = a.iter().map(|o| o.content).collect();
        assert_eq!(v_contents, aligned_contents(tail_from, 40, 7000));
        assert_eq!(a_contents, aligned_contents(tail_from, 20, 7000));

        let mut shifts = Vec::new();
        for o in v.iter().chain(a.iter()) {
            shifts.push(o.output_pts - o.input_pts);
        }
        let min = *shifts.iter().min().unwrap();
        let max = *shifts.iter().max().unwrap();
        assert!(
            max - min <= 1,
            "tail shift spread {}..{}",
            min,
            max
        );

        for frame in &v {
            let nearest = a.iter().min_by_key(|o| (o.output_pts - frame.output_pts).abs()).unwrap();
            assert!(
                (nearest.content - frame.content).abs() <= 20,
                "video content {} at {} vs audio content {}",
                frame.content,
                frame.output_pts,
                nearest.content
            );
        }
    }

    #[test]
    fn broken_then_good_slew_follow_matches_on_one_shift() {
        let mut played = play(Convergence::SlewFollow);
        assert_monotonic_grid(&played.video);
        assert_monotonic_grid(&played.audio);
        let mut skewed = false;
        for frame in &played.video {
            if !(1000..2500).contains(&frame.output_pts) {
                continue;
            }
            if let Some(audio) = played
                .audio
                .iter()
                .min_by_key(|o| (o.output_pts - frame.output_pts).abs())
            {
                if (audio.content - frame.content).abs() > 20 {
                    skewed = true;
                    break;
                }
            }
        }
        assert!(skewed, "broken window never separated audio and video content");
        assert_tail(&played, 5000);
        assert_timeout_fill(&mut played);
    }

    #[test]
    fn broken_then_good_snap_matches_after_the_recovering_frame() {
        let mut played = play(Convergence::Snap);
        assert_monotonic_grid(&played.video);
        assert_monotonic_grid(&played.audio);
        assert_tail(&played, 2540);
        assert_timeout_fill(&mut played);
    }
}
