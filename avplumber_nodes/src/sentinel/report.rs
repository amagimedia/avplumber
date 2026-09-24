//! Timeshift history: a packed file, a text file, and an HTTP POST.
//!
//! Wallclock tracking reports drift. It does not move the output timestamps.

use std::fs::OpenOptions;
use std::io::Write;
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use avplumber_f7k::graph::timebase::MILLISECONDS;
use avplumber_f7k::graph::timestamp::{Ts, TsDelta};

pub struct Reporter {
    history: Mutex<Option<std::fs::File>>,
    history_text: Mutex<Option<std::fs::File>>,
    url: Option<String>,
    track: bool,
    max_drift: Duration,
    grace: Duration,
    interval: Option<Duration>,
    state: Mutex<ReportState>,
}

struct ReportState {
    first: bool,
    next_report: Option<Instant>,
    wall_offset_ms: Option<i64>,
    drifted_since: Option<Instant>,
}

impl Reporter {
    pub fn open(
        history: Option<&str>,
        history_text: Option<&str>,
        url: Option<String>,
        track: bool,
        max_drift_s: f64,
        grace_s: f64,
        interval_s: Option<f64>,
    ) -> Result<Self, String> {
        Ok(Self {
            history: Mutex::new(match history {
                Some(path) => Some(open_append(path)?),
                None => None,
            }),
            history_text: Mutex::new(match history_text {
                Some(path) => Some(open_append(path)?),
                None => None,
            }),
            url,
            track,
            max_drift: Duration::from_secs_f64(max_drift_s.max(0.0)),
            grace: Duration::from_secs_f64(grace_s.max(0.0)),
            interval: interval_s.map(|s| Duration::from_secs_f64(s.max(0.0))),
            state: Mutex::new(ReportState {
                first: true,
                next_report: None,
                wall_offset_ms: None,
                drifted_since: None,
            }),
        })
    }

    /// A shift change. Offsets are milliseconds relative to `start`, and the
    /// first record is forced to `changed_at = 0`, as the C++ history is.
    pub fn shift_changed(&self, changed_at: Ts, shift: TsDelta, start: Ts) {
        if !self.due() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let mut at_ms = to_ms(changed_at) - to_ms(start);
        if state.first {
            at_ms = 0;
            state.first = false;
        }
        let shift_ms = to_ms_delta(shift) - to_ms(start);
        let wall_ms = state.wall_offset_ms.unwrap_or(0);
        let start_ms = to_ms(start);
        drop(state);
        self.write(at_ms, shift_ms, wall_ms, start_ms);
    }

    /// Compare the output timestamp with the wall clock. A drift that stays
    /// past `max_wallclock_drift` for the grace period is recorded and not
    /// corrected.
    pub fn observe_wallclock(&self, output: Ts, start: Ts) {
        if !self.track || !output.is_valid() {
            return;
        }
        let now_ms = unix_ms();
        let new_offset = to_ms(output) - now_ms;
        let mut state = self.state.lock().unwrap();
        let report = match state.wall_offset_ms {
            None => true,
            Some(prev) => {
                let diff_ms = (new_offset - prev).abs();
                if diff_ms > self.max_drift.as_millis() as i64 {
                    match state.drifted_since {
                        None => {
                            state.drifted_since = Some(Instant::now());
                            false
                        }
                        Some(since) => since.elapsed() >= self.grace,
                    }
                } else {
                    state.drifted_since = None;
                    false
                }
            }
        };
        if !report {
            return;
        }
        state.wall_offset_ms = Some(new_offset);
        state.drifted_since = None;
        let first = state.first;
        if first {
            state.first = false;
        }
        drop(state);
        log::info!(
            "wallclock drift, offset {new_offset} ms (reporting only, timestamps unchanged)"
        );
        let at_ms = if first { 0 } else { to_ms(output) - to_ms(start) };
        self.write(at_ms, 0, new_offset, to_ms(start));
    }

    fn due(&self) -> bool {
        let Some(interval) = self.interval else {
            return true;
        };
        let mut state = self.state.lock().unwrap();
        if let Some(next) = state.next_report
            && Instant::now() < next
        {
            return false;
        }
        state.next_report = Some(Instant::now() + interval);
        true
    }

    fn write(&self, changed_at: i64, input_pts_offset: i64, wallclock_offset: i64, output_pts_offset: i64) {
        if let Some(file) = self.history.lock().unwrap().as_mut() {
            for value in [changed_at, input_pts_offset, wallclock_offset, output_pts_offset] {
                let _ = file.write_all(&value.to_ne_bytes());
            }
            let _ = file.flush();
        }
        if let Some(file) = self.history_text.lock().unwrap().as_mut() {
            let _ = writeln!(
                file,
                "{changed_at} {input_pts_offset} {wallclock_offset} {output_pts_offset}"
            );
            let _ = file.flush();
        }
        if let Some(url) = &self.url {
            let body = format!(
                "{{\"changed_at\":{changed_at},\"input_pts_offset\":{input_pts_offset},\"wallclock_offset\":{wallclock_offset},\"output_pts_offset\":{output_pts_offset}}}"
            );
            post_json(url, &body);
        }
    }
}

fn open_append(path: &str) -> Result<std::fs::File, String> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("history file {path}: {e}"))
}

fn to_ms(ts: Ts) -> i64 {
    if !ts.is_valid() {
        return 0;
    }
    ts.rescale(MILLISECONDS).ticks()
}

fn to_ms_delta(delta: TsDelta) -> i64 {
    if !delta.is_valid() {
        return 0;
    }
    delta.rescale(MILLISECONDS).ticks()
}

fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Best-effort POST. A dead receiver must not fail the node.
pub fn post_json(url: &str, body: &str) {
    let Some(rest) = url.strip_prefix("http://") else {
        log::warn!("reporting_url `{url}` is not http://");
        return;
    };
    let (hostport, path) = match rest.split_once('/') {
        Some((host, path)) => (host, format!("/{path}")),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match hostport.split_once(':') {
        Some((host, port)) => {
            let Ok(port) = port.parse::<u16>() else {
                log::warn!("reporting_url `{url}` has no port");
                return;
            };
            (host, port)
        }
        None => (hostport, 80),
    };
    let address = format!("{host}:{port}");
    let Ok(mut stream) = TcpStream::connect_timeout(
        &address
            .parse()
            .unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], port))),
        Duration::from_millis(200),
    ) else {
        log::warn!("reporting_url `{url}` is not reachable");
        return;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
    let request = format!(
        "POST {path} HTTP/1.0\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(request.as_bytes());
}
