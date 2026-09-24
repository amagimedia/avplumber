//! JSON fields shared by `sentinel_video` and `resample_audio`.

use avplumber_f7k::graph::media::AvpRational;
use avplumber_f7k::graph::timestamp::Ts;
use avplumber_f7k::services::correction::{Convergence, CorrectionGroup, GroupPolicy};
use avplumber_f7k::util::parse_iso8601_ms;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct CorrectionParams {
    #[serde(default = "default_group")]
    pub correction_group: String,
    #[serde(default = "default_mode")]
    pub convergence: String,
    #[serde(default = "default_slew")]
    pub max_slew: f64,
    /// Seconds. A jump larger than this rebases.
    #[serde(default = "default_rebase")]
    pub rebase_threshold: f64,
    /// Seconds. Shifts closer than this are left alone.
    #[serde(default = "default_deadband")]
    pub max_streams_diff: f64,
    #[serde(default)]
    pub anchor_member: Option<String>,
    #[serde(default)]
    pub lock_timeshift: bool,
    #[serde(default)]
    pub forward_start_shift: bool,
    /// Seconds. Absent means 10, matching the C++ corrector.
    #[serde(default)]
    pub start_ts: Option<f64>,
    /// Seconds without a frame before backup media is considered.
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    #[serde(default)]
    pub eof_passthrough: bool,
    #[serde(default)]
    pub track_wallclock: bool,
    #[serde(default = "default_wall_drift")]
    pub max_wallclock_drift: f64,
    #[serde(default = "default_grace")]
    pub wallclock_drift_grace_period: f64,
    #[serde(default)]
    pub history_file: Option<String>,
    #[serde(default)]
    pub history_file_text: Option<String>,
    #[serde(default)]
    pub history_report_interval: Option<f64>,
    #[serde(default)]
    pub reporting_url: Option<String>,
    #[serde(default)]
    pub hold_until_ms: Option<i64>,
    #[serde(default)]
    pub hold_until_iso: Option<String>,
}

fn default_group() -> String {
    String::new()
}
fn default_mode() -> String {
    "slew".into()
}
fn default_slew() -> f64 {
    0.05
}
fn default_rebase() -> f64 {
    0.5
}
fn default_deadband() -> f64 {
    0.001
}
fn default_timeout() -> f64 {
    1.0
}
fn default_wall_drift() -> f64 {
    0.04
}
fn default_grace() -> f64 {
    1.0
}

impl Default for CorrectionParams {
    fn default() -> Self {
        Self {
            correction_group: default_group(),
            convergence: default_mode(),
            max_slew: default_slew(),
            rebase_threshold: default_rebase(),
            max_streams_diff: default_deadband(),
            anchor_member: None,
            lock_timeshift: false,
            forward_start_shift: false,
            start_ts: None,
            timeout: default_timeout(),
            eof_passthrough: false,
            track_wallclock: false,
            max_wallclock_drift: default_wall_drift(),
            wallclock_drift_grace_period: default_grace(),
            history_file: None,
            history_file_text: None,
            history_report_interval: None,
            reporting_url: None,
            hold_until_ms: None,
            hold_until_iso: None,
        }
    }
}

impl CorrectionParams {
    pub fn mode(&self) -> Result<Convergence, String> {
        match self.convergence.as_str() {
            "slew" => Ok(Convergence::Slew),
            "slew_follow" => Ok(Convergence::SlewFollow),
            "anchor" => Ok(Convergence::Anchor),
            "snap" => Ok(Convergence::Snap),
            "drop_fill" => Ok(Convergence::DropFill),
            other => Err(format!(
                "unknown convergence `{other}` (slew, slew_follow, anchor, snap, drop_fill)"
            )),
        }
    }

    pub fn policy(&self) -> Result<GroupPolicy, String> {
        Ok(GroupPolicy {
            mode: self.mode()?,
            max_slew: self.max_slew,
            rebase_threshold: self.rebase_threshold,
            deadband: self.max_streams_diff,
            anchor: self.anchor_member.clone(),
            locked: self.lock_timeshift,
        })
    }

    /// Seconds, or 10 when the script left `start_ts` out.
    pub fn start_seconds(&self) -> f64 {
        self.start_ts.unwrap_or(10.0)
    }

    pub fn hold_until_unix_ms(&self) -> Result<Option<i64>, String> {
        if let Some(ms) = self.hold_until_ms {
            return Ok(Some(ms));
        }
        match &self.hold_until_iso {
            Some(text) => parse_iso8601_ms(text)
                .map(Some)
                .map_err(|err| format!("hold_until_iso {err}")),
            None => Ok(None),
        }
    }
}

/// Set the group timebase, the mode, and `start_ts`. The returned start is in
/// the group's timebase (which may be coarser than `frame_tb`).
pub fn configure_member(
    group: &CorrectionGroup,
    params: &CorrectionParams,
    frame_tb: AvpRational,
) -> Result<Ts, String> {
    group.set_output_tb(frame_tb);
    let tb = group.output_tb();
    group.configure(params.policy()?)?;
    let start = Ts::new(seconds_to_ticks(params.start_seconds(), tb), tb);
    group.set_start_ts(start);
    Ok(start)
}

pub fn seconds_to_ticks(seconds: f64, tb: AvpRational) -> i64 {
    if tb.num <= 0 || tb.den <= 0 {
        return 1;
    }
    (seconds * tb.den as f64 / tb.num as f64).round() as i64
}
