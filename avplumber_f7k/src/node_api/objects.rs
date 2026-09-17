//! Control-plane knobs (`node.object.set` / `get`).
//!
//! Independent of scheduling (blocking / poll / async) and of the grain
//! in/out adapters ([`SisoNode`](crate::node_api::SisoNode),
//! [`PollInput`](crate::node_api::PollInput),
//! [`InputHandler`](crate::node_api::InputHandler)). A node that has keys
//! implements [`NodeObjects`] and returns `Some(self)` from the body's
//! `objects()` hook so the [`Node`](crate::graph::node::Node) wrapper can
//! dispatch. A node that has no keys implements nothing.
//!
//! Rust will not pick up `impl NodeObjects for T` from a generic wrapper
//! (`Polling<T>`, `SisoPollAdapter<T>`, …). The `objects()` hook is that
//! missing link, not a second copy of set/get.

use serde_json::Value;

/// Named knobs the control thread can turn while the body runs.
///
/// Must not take a lock the body holds across a wait. Implement this on the
/// author type (`ForceKeyframe`, `Realtime`, …), not on an adapter.
pub trait NodeObjects: Send + Sync {
    fn set_object(&self, key: &str, _value: &Value) -> Result<(), String> {
        Err(format!("no object `{key}` to set"))
    }

    fn get_object(&self, key: &str) -> Result<Value, String> {
        Err(format!("no object `{key}` to get"))
    }
}

pub(crate) fn missing_set(node_name: &str, key: &str) -> String {
    format!("{node_name} has no object `{key}` to set")
}

pub(crate) fn missing_get(node_name: &str, key: &str) -> String {
    format!("{node_name} has no object `{key}` to get")
}

pub(crate) fn set_on(
    objects: Option<&dyn NodeObjects>,
    node_name: &str,
    key: &str,
    value: &Value,
) -> Result<(), String> {
    match objects {
        Some(o) => o.set_object(key, value),
        None => Err(missing_set(node_name, key)),
    }
}

pub(crate) fn get_on(
    objects: Option<&dyn NodeObjects>,
    node_name: &str,
    key: &str,
) -> Result<Value, String> {
    match objects {
        Some(o) => o.get_object(key),
        None => Err(missing_get(node_name, key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Has;

    impl NodeObjects for Has {
        fn get_object(&self, key: &str) -> Result<Value, String> {
            match key {
                "k" => Ok(json!(1)),
                other => Err(format!("no `{other}`")),
            }
        }
    }

    #[test]
    fn set_on_none_names_the_node() {
        let err = set_on(None, "sink", "k", &json!(true)).unwrap_err();
        assert!(err.contains("sink") && err.contains("k"), "got {err}");
    }

    #[test]
    fn get_on_reaches_the_trait() {
        assert_eq!(get_on(Some(&Has), "n", "k").unwrap(), json!(1));
        assert!(get_on(Some(&Has), "n", "x").is_err());
    }
}
