//! Node parameters → `AVDictionary` (C++ `parametersToDict`), plus reporting
//! the entries libav did not take.

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use rsmpeg::avutil::AVDictionary;
use rusty_ffmpeg::ffi;
use serde_json::Value;

/// An option dictionary libav is allowed to consume entries from.
///
/// Not [`rsmpeg::avutil::AVDictionary`]: that is a non-null handle whose `set`
/// consumes `self`, while the `avformat_open_input(…, &mut options)` protocol
/// rewrites the caller's pointer and leaves the *unconsumed* entries behind for
/// the caller to report. So this owns the raw pointer instead.
pub struct Options {
    raw: *mut ffi::AVDictionary,
}

// Plain key/value data with no thread affinity; libav only touches it through
// the pointer we hand it. Needed because node state lives behind a `Mutex`.
unsafe impl Send for Options {}

impl Options {
    pub fn new() -> Self {
        Self {
            raw: std::ptr::null_mut(),
        }
    }

    /// `{"key": value}` → dictionary, stringifying non-string values exactly as
    /// C++ `parametersToDict` does. Absent, `null` and `{}` are all empty.
    pub fn from_json(params: Option<&Value>) -> Result<Self, String> {
        let mut out = Self::new();
        let Some(params) = params else {
            return Ok(out);
        };
        match params {
            Value::Null => Ok(out),
            Value::Object(map) => {
                for (key, value) in map {
                    let value = match value {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    out.set(key, &value)?;
                }
                Ok(out)
            }
            _ => Err("options expects an object of \"key\": value pairs".into()),
        }
    }

    /// The same from ready-made pairs, for the per-stream `metadata` a
    /// [`MuxStream`](crate::graph::spec::MuxStream) carries.
    pub fn from_pairs(pairs: &[(String, String)]) -> Result<Self, String> {
        let mut out = Self::new();
        for (key, value) in pairs {
            out.set(key, value)?;
        }
        Ok(out)
    }

    /// Hand the entries over to rsmpeg, for the calls that want an owned
    /// [`AVDictionary`] instead of a pointer (`write_header`, `set_metadata`, the
    /// output-context builder). `None` is how all of them read "no options".
    pub fn into_av_dictionary(mut self) -> Option<AVDictionary> {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        NonNull::new(raw).map(|raw| unsafe { AVDictionary::from_raw(raw) })
    }

    /// The way back: what such a call left unconsumed, ready to be reported by
    /// [`Options::warn_leftovers`].
    pub fn from_av_dictionary(dict: Option<AVDictionary>) -> Self {
        Self {
            raw: dict
                .map(|dict| dict.into_raw().as_ptr())
                .unwrap_or(std::ptr::null_mut()),
        }
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let c_key = CString::new(key).map_err(|_| format!("option key `{key}` contains a NUL"))?;
        let c_value =
            CString::new(value).map_err(|_| format!("option `{key}` value contains a NUL"))?;
        let ret = unsafe { ffi::av_dict_set(&mut self.raw, c_key.as_ptr(), c_value.as_ptr(), 0) };
        if ret < 0 {
            return Err(format!("av_dict_set failed for `{key}`"));
        }
        Ok(())
    }

    /// The `*mut *mut AVDictionary` the consuming libav calls want.
    pub fn as_mut_ptr(&mut self) -> *mut *mut ffi::AVDictionary {
        &mut self.raw
    }

    pub fn is_empty(&self) -> bool {
        unsafe { ffi::av_dict_count(self.raw) == 0 }
    }

    /// Entries libav did not consume: a typo in a script, or an option this
    /// codec/muxer does not know.
    pub fn leftovers(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        if self.raw.is_null() {
            return out;
        }
        let mut entry: *const ffi::AVDictionaryEntry = std::ptr::null();
        loop {
            entry = unsafe {
                ffi::av_dict_get(
                    self.raw,
                    c"".as_ptr(),
                    entry,
                    ffi::AV_DICT_IGNORE_SUFFIX as i32,
                )
            };
            if entry.is_null() {
                break;
            }
            let key = unsafe { CStr::from_ptr((*entry).key) };
            let value = unsafe { CStr::from_ptr((*entry).value) };
            out.push((
                key.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            ));
        }
        out
    }

    /// Like C++: log and carry on, since an unknown option is not fatal.
    pub fn warn_leftovers(&self, node: &str, consumer: &str) {
        let left = self.leftovers();
        if left.is_empty() {
            return;
        }
        let list = left
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ");
        log::warn!("{node}: {consumer} ignored these options: {list}");
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Options {
    fn drop(&mut self) {
        unsafe { ffi::av_dict_free(&mut self.raw) };
    }
}
