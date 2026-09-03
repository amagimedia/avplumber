//! Line-based control language (script-compatible). TCP is deferred.

use std::path::Path;

use serde_json::Value;

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

fn group_status(core: &Instance, rest: &str) -> Result<String, String> {
    let status = core.group_status(rest.trim()).map_err(|e| e.to_string())?;
    serde_json::to_string(&status).map_err(|e| e.to_string())
}
