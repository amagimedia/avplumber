//! PTS-keyed SharedTimeline store.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::graph::AvpRational;
use crate::graph::timebase::{self, MILLISECONDS};

#[derive(Clone, Default)]
struct Channel {
    keys: HashMap<String, BTreeMap<i64, Option<String>>>,
}

pub trait SharedTimeline: Send + Sync {
    fn set(&self, channel: &str, key: &str, at_pts_ms: i64, value_json: &str);
    fn clear_key(&self, channel: &str, key: &str);
    fn gc(&self, before_pts_ms: i64);
    fn get(&self, channel: &str, key: &str, frame_pts: i64, tb: AvpRational) -> String;
    fn get_opt(&self, channel: &str, key: &str, frame_pts: i64, tb: AvpRational) -> Option<String>;
}

pub struct InMemoryTimeline {
    inner: Mutex<HashMap<String, Channel>>,
}

impl InMemoryTimeline {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryTimeline {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedTimeline for InMemoryTimeline {
    fn set(&self, channel: &str, key: &str, at_pts_ms: i64, value_json: &str) {
        let mut g = self.inner.lock().unwrap();
        g.entry(channel.to_string())
            .or_default()
            .keys
            .entry(key.to_string())
            .or_default()
            .insert(at_pts_ms, Some(value_json.to_string()));
    }
    fn clear_key(&self, channel: &str, key: &str) {
        if let Some(ch) = self.inner.lock().unwrap().get_mut(channel) {
            ch.keys.remove(key);
        }
    }
    fn gc(&self, before_pts_ms: i64) {
        let mut g = self.inner.lock().unwrap();
        for ch in g.values_mut() {
            for map in ch.keys.values_mut() {
                let keep = map.split_off(&before_pts_ms);
                let last_before = map.iter().next_back().map(|(k, v)| (*k, v.clone()));
                *map = keep;
                if let Some((k, v)) = last_before {
                    map.insert(k, v);
                }
            }
        }
    }
    fn get(&self, channel: &str, key: &str, frame_pts: i64, tb: AvpRational) -> String {
        self.get_opt(channel, key, frame_pts, tb)
            .unwrap_or_default()
    }
    fn get_opt(&self, channel: &str, key: &str, frame_pts: i64, tb: AvpRational) -> Option<String> {
        let pts_ms = timebase::rescale(frame_pts, tb, MILLISECONDS);
        let g = self.inner.lock().unwrap();
        let ch = g.get(channel)?;
        let map = ch.keys.get(key)?;
        map.range(..=pts_ms)
            .next_back()
            .and_then(|(_, v)| v.clone())
    }
}

pub struct TimelineService {
    groups: Mutex<HashMap<String, Arc<InMemoryTimeline>>>,
}

impl TimelineService {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
        }
    }
    pub fn get_or_create(&self, name: &str) -> Arc<InMemoryTimeline> {
        self.groups
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(InMemoryTimeline::new()))
            .clone()
    }
}

impl Default for TimelineService {
    fn default() -> Self {
        Self::new()
    }
}
