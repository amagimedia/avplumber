//! Line-based control language (script-compatible); [`tcp`] serves it on a
//! socket in the C++ wire format.

pub mod tcp;

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use crate::services::playback::{Playback, Target};

use crate::factory::NodeEnvelope;
use crate::{CoreError, EdgeKind, Instance, NodePads, NodeRequest, PadDirection};

#[derive(Debug)]
pub struct ScriptError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ScriptError {}

pub fn exec_file(inst: &Instance, path: &Path) -> Result<(), ScriptError> {
    let text = std::fs::read_to_string(path).map_err(|e| ScriptError {
        line: 0,
        message: e.to_string(),
    })?;
    exec_script(inst, &text)
}

pub fn exec_script(inst: &Instance, text: &str) -> Result<(), ScriptError> {
    for (i, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        exec_line(inst, line).map_err(|message| ScriptError {
            line: i + 1,
            message,
        })?;
    }
    Ok(())
}

fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

pub fn exec_line(core: &Instance, line: &str) -> Result<String, String> {
    let line = strip_comment(line).trim();
    if line.is_empty() {
        return Ok(String::new());
    }
    let (cmd, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    match cmd {
        "node.add" => node_add(core, rest),
        "edge.add" => edge_add(core, rest),
        "queue.plan_capacity" => plan_capacity(core, rest),
        "group.start" => group_cmd(core, rest, true),
        "group.stop" => group_cmd(core, rest, false),
        "group.restart" => group_restart(core, rest),
        "group.status" => group_status(core, rest),
        "node.object.set" => node_object_set(core, rest),
        "node.object.get" => node_object_get(core, rest),
        "seek" => seek(core, rest),
        "pause" => pause(core, rest),
        "resume" => resume(core, rest),
        "speed.set" => speed_set(core, rest),
        "speed.get" => speed_get(core, rest),
        "playback.status" => playback_status(core, rest),
        "hello" | "version" => Ok("ok".into()),
        "bye" => Ok("bye".into()),
        _ => Err(format!("unknown command {cmd}")),
    }
}

fn json_strings(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn node_add(core: &Instance, rest: &str) -> Result<String, String> {
    let mut v: Value = serde_json::from_str(rest).map_err(|e| e.to_string())?;
    let obj = v.as_object_mut().ok_or("node.add expects a JSON object")?;
    // Nested params (old stub) flatten into the object.
    if let Some(Value::Object(p)) = obj.remove("params") {
        for (k, val) in p {
            obj.entry(k).or_insert(val);
        }
    }
    let env = NodeEnvelope::extract(obj)?;
    // Put the envelope keys back, exactly like `NodeRequest::from_json` does for
    // the C ABI: a node type may need them (`mux` numbers its output streams by
    // the order of `src`), and both entry points must produce the same canonical
    // params, since that string is what a reconstruction rebuilds from.
    if let Some(group) = env.group.as_ref() {
        obj.insert("group".into(), Value::String(group.clone()));
    }
    if let Some(src) = env.src.as_ref() {
        obj.insert("src".into(), src.clone());
    }
    if let Some(dst) = env.dst.as_ref() {
        obj.insert("dst".into(), dst.clone());
    }
    let remainder = Value::Object(obj.clone());
    let name = env.name.clone();
    let group = env.group.clone();
    let src = env.src.clone();
    let dst = env.dst.clone();
    let request = NodeRequest::from_envelope(env, remainder).map_err(|e| e.to_string())?;
    if group.is_none()
        && (request.restart == Some(crate::RestartPolicy::RestartGroup)
            || request.on_error == Some(crate::RestartPolicy::RestartGroup))
    {
        return Err("RestartGroup policy requires group membership".into());
    }
    let node = core.create_node(request).map_err(|e| e.to_string())?;

    let mut created_group = None;
    let configure = (|| {
        bind_named(core, &name, PadDirection::Input, src.as_ref(), &node.pads)?;
        bind_named(core, &name, PadDirection::Output, dst.as_ref(), &node.pads)?;
        if let Some(group_name) = group.as_deref() {
            match core.create_group(group_name) {
                Ok(()) => created_group = Some(group_name),
                Err(CoreError::AlreadyExists { .. }) => {}
                Err(error) => return Err(error.to_string()),
            }
            core.add_group_member(group_name, &name)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    if let Err(error) = configure {
        let _ = core.destroy_node(&name);
        if let Some(group_name) = created_group {
            let _ = core.destroy_group(group_name);
        }
        return Err(error);
    }
    Ok("ok".into())
}

fn bind_named(
    core: &Instance,
    node: &str,
    direction: PadDirection,
    value: Option<&Value>,
    pads: &NodePads,
) -> Result<(), String> {
    let Some(v) = value else { return Ok(()) };
    let names = json_strings(v);
    let declared = match direction {
        PadDirection::Input => &pads.sources,
        PadDirection::Output => &pads.sinks,
    };
    for (i, edge_name) in names.iter().enumerate() {
        let pad = if declared.len() == names.len() {
            declared[i].name.as_str()
        } else if declared.len() == 1 {
            declared[0].name.as_str()
        } else {
            edge_name.as_str()
        };
        core.bind_edge(node, pad, direction, edge_name)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn edge_add(core: &Instance, rest: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(rest).map_err(|e| e.to_string())?;
    let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("e");
    let from = v
        .get("from")
        .and_then(|x| x.as_str())
        .ok_or("missing from")?;
    let to = v.get("to").and_then(|x| x.as_str()).ok_or("missing to")?;
    let (pn, pp) = from.split_once('.').ok_or("from must be node.pad")?;
    let (cn, cp) = to.split_once('.').ok_or("to must be node.pad")?;
    core.connect_edge(name, pn, pp, cn, cp, EdgeKind::default())
        .map_err(|e| e.to_string())?;
    Ok("ok".into())
}

fn plan_capacity(core: &Instance, rest: &str) -> Result<String, String> {
    let mut parts = rest.split_whitespace();
    let name = parts.next().ok_or("queue.plan_capacity <name> <n>")?;
    let n: usize = parts
        .next()
        .ok_or("missing capacity")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    core.plan_capacity(name, n);
    Ok("ok".into())
}

fn group_cmd(core: &Instance, rest: &str, start: bool) -> Result<String, String> {
    let name = rest.trim();
    if start {
        core.start_group(name).map_err(|e| e.to_string())?;
    } else {
        core.stop_group(name).map_err(|e| e.to_string())?;
    }
    Ok("ok".into())
}

fn group_restart(core: &Instance, rest: &str) -> Result<String, String> {
    core.restart_group(rest.trim()).map_err(|e| e.to_string())?;
    Ok("ok".into())
}

/// `node.object.set <node> <key> <json>`.
fn node_object_set(core: &Instance, rest: &str) -> Result<String, String> {
    let mut parts = rest.trim().splitn(3, char::is_whitespace);
    let node = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("node.object.set <node> <key> <json>")?;
    let key = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or("missing object key")?;
    let value: Value = match parts.next().map(str::trim) {
        Some(text) if !text.is_empty() => serde_json::from_str(text).map_err(|e| e.to_string())?,
        _ => Value::Null,
    };
    let instance = core
        .node(node)
        .ok_or_else(|| format!("unknown node {node}"))?;
    instance.node.set_object(key, &value)?;
    Ok("ok".into())
}

/// `node.object.get <node> <key>`: the value as JSON.
fn node_object_get(core: &Instance, rest: &str) -> Result<String, String> {
    let mut parts = rest.split_whitespace();
    let node = parts.next().ok_or("node.object.get <node> <key>")?;
    let key = parts.next().ok_or("missing object key")?;
    let instance = core
        .node(node)
        .ok_or_else(|| format!("unknown node {node}"))?;
    let value = instance.node.get_object(key)?;
    serde_json::to_string(&value).map_err(|e| e.to_string())
}

/// The playback group a verb addresses; created on demand, so a group that
/// only has a pacing node still answers `pause` and `speed.set`.
fn playback(core: &Instance, name: &str) -> Result<Arc<Playback>, String> {
    if name.is_empty() {
        return Err("missing playback group name".into());
    }
    Ok(core.services.playback(name))
}

/// `seek <group> now <target>` | `frame <N|+N|-N>` | `at <when> <target>` |
/// `clear` | `live` | `end`. Targets: `12000`, `+500`, `01:02:03.250`,
/// `2026-08-10T12:00:00.000`. Replies with the media time seeked to.
fn seek(core: &Instance, rest: &str) -> Result<String, String> {
    let mut parts = rest.split_whitespace();
    let group = parts
        .next()
        .ok_or("seek <group> now|frame|at|clear|live|end ...")?;
    let verb = parts.next().ok_or("seek: missing subcommand")?;
    let playback = playback(core, group)?;
    match verb {
        "now" => {
            let target = Target::parse(parts.next().ok_or("seek now: missing target")?)?;
            playback.seek(target).map(|ms| ms.to_string())
        }
        "frame" => {
            let text = parts.next().ok_or("seek frame: missing frame number")?;
            let number: i64 = text
                .parse()
                .map_err(|_| format!("`{text}` is not a frame number"))?;
            let target = if text.starts_with(['+', '-']) {
                Target::RelativeFrames(number)
            } else {
                Target::Frame(number)
            };
            playback.seek(target).map(|ms| ms.to_string())
        }
        "at" => {
            let when = Target::parse(parts.next().ok_or("seek at: missing when")?)?;
            let target = Target::parse(parts.next().ok_or("seek at: missing target")?)?;
            playback.seek_at(when, target).map(|()| "ok".into())
        }
        "clear" => {
            playback.clear_scheduled();
            Ok("ok".into())
        }
        "live" => playback.seek(Target::Live).map(|ms| ms.to_string()),
        "end" => playback.seek(Target::End).map(|ms| ms.to_string()),
        other => Err(format!("seek: unknown subcommand `{other}`")),
    }
}

/// `pause <group> now` | `pause <group> at <target>`.
fn pause(core: &Instance, rest: &str) -> Result<String, String> {
    let mut parts = rest.split_whitespace();
    let group = parts.next().ok_or("pause <group> now|at <target>")?;
    let playback = playback(core, group)?;
    match parts.next().unwrap_or("now") {
        "now" => {
            playback.pause();
            Ok("ok".into())
        }
        "at" => {
            let target = Target::parse(parts.next().ok_or("pause at: missing target")?)?;
            playback.pause_at(target).map(|()| "ok".into())
        }
        other => Err(format!("pause: unknown subcommand `{other}`")),
    }
}

fn resume(core: &Instance, rest: &str) -> Result<String, String> {
    let group = rest.split_whitespace().next().ok_or("resume <group>")?;
    playback(core, group)?.resume();
    Ok("ok".into())
}

/// `speed.set <group> <rate>`: negative plays backwards, zero pauses.
fn speed_set(core: &Instance, rest: &str) -> Result<String, String> {
    let mut parts = rest.split_whitespace();
    let group = parts.next().ok_or("speed.set <group> <rate>")?;
    let text = parts.next().ok_or("speed.set: missing rate")?;
    let rate: f64 = text
        .parse()
        .map_err(|_| format!("`{text}` is not a rate"))?;
    playback(core, group)?.set_rate(rate).map(|()| "ok".into())
}

fn speed_get(core: &Instance, rest: &str) -> Result<String, String> {
    let group = rest.split_whitespace().next().ok_or("speed.get <group>")?;
    Ok(playback(core, group)?.rate().to_string())
}

fn playback_status(core: &Instance, rest: &str) -> Result<String, String> {
    let group = rest
        .split_whitespace()
        .next()
        .ok_or("playback.status <group>")?;
    serde_json::to_string(&playback(core, group)?.status()).map_err(|e| e.to_string())
}

fn group_status(core: &Instance, rest: &str) -> Result<String, String> {
    let status = core.group_status(rest.trim()).map_err(|e| e.to_string())?;
    serde_json::to_string(&status).map_err(|e| e.to_string())
}
