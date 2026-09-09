//! Named hardware devices. Port of C++ `src/hwaccel.hpp` plus the
//! `InstanceSharedObjects<HWAccelDevice>` registry it lived in.
//!
//! One `AVHWDeviceContext` per name, created by `hwaccel.init` and looked up by
//! the codecs that name it. Everything else — the frame pool, the format
//! negotiation, the surfaces themselves — belongs to libavcodec; this service
//! only owns the device and hands out references to it.
//!
//! Frames stay on the device because nothing in the graph copies them: a
//! hardware `AVFrame` carries its own `hw_frames_ctx`, edges move it by
//! reference, and a decoder and an encoder that name the same device speak
//! about the same GPU. The one thing the frame does not carry to a node that
//! has to open a codec *before* seeing a frame is which software format the
//! surface holds, which is why [`Spec::Video`](crate::graph::spec::Spec) has
//! `sw_pix_fmt`.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use rsmpeg::avutil::{AVHWDeviceContext, hwdevice_find_type_by_name};
use rusty_ffmpeg::ffi;
use serde_json::Value;

use crate::libav::dict::Options;

/// One initialized device, shared by every node that names it.
pub struct HwDevice {
    /// The name it was registered under, for logs.
    pub name: String,
    /// `AV_HWDEVICE_TYPE_CUDA` and friends.
    pub device_type: ffi::AVHWDeviceType,
    ctx: AVHWDeviceContext,
}

impl std::fmt::Debug for HwDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HwDevice({}, {})",
            self.name,
            type_name(self.device_type)
        )
    }
}

// The context is an immutable, atomically refcounted `AVBufferRef` after
// creation; libav's own users (codecs on their own threads) share it exactly
// this way. Nothing here mutates it, and `device_ref` only takes a reference.
unsafe impl Send for HwDevice {}
unsafe impl Sync for HwDevice {}

impl HwDevice {
    /// A new reference for a codec context to own, as C++ `refDeviceContext`.
    pub fn device_ref(&self) -> AVHWDeviceContext {
        self.ctx.clone()
    }

    /// Borrowed, for the calls that take their own reference internally
    /// (`av_hwframe_ctx_alloc`). C++ `deviceContext`.
    pub fn as_ptr(&self) -> *mut ffi::AVBufferRef {
        self.ctx.as_ptr() as *mut ffi::AVBufferRef
    }

    /// The pixel format a surface of this device carries, C++
    /// `hardwarePixelFormat`. `AV_PIX_FMT_NONE` for a device type whose frames
    /// this build has no format for.
    pub fn hw_pix_fmt(&self) -> ffi::AVPixelFormat {
        match self.device_type {
            ffi::AV_HWDEVICE_TYPE_CUDA => ffi::AV_PIX_FMT_CUDA,
            ffi::AV_HWDEVICE_TYPE_VAAPI => ffi::AV_PIX_FMT_VAAPI,
            ffi::AV_HWDEVICE_TYPE_QSV => ffi::AV_PIX_FMT_QSV,
            ffi::AV_HWDEVICE_TYPE_DRM => ffi::AV_PIX_FMT_DRM_PRIME,
            ffi::AV_HWDEVICE_TYPE_VDPAU => ffi::AV_PIX_FMT_VDPAU,
            ffi::AV_HWDEVICE_TYPE_OPENCL => ffi::AV_PIX_FMT_OPENCL,
            ffi::AV_HWDEVICE_TYPE_VULKAN => ffi::AV_PIX_FMT_VULKAN,
            ffi::AV_HWDEVICE_TYPE_VIDEOTOOLBOX => ffi::AV_PIX_FMT_VIDEOTOOLBOX,
            ffi::AV_HWDEVICE_TYPE_D3D11VA => ffi::AV_PIX_FMT_D3D11,
            _ => ffi::AV_PIX_FMT_NONE,
        }
    }

    /// A frame pool on this device for `width`x`height` surfaces of
    /// `sw_pix_fmt`, initialized and ready to hand to a codec context. C++ does
    /// this inline in `openEncoder`.
    pub fn frames_ctx(
        &self,
        width: i32,
        height: i32,
        sw_pix_fmt: ffi::AVPixelFormat,
    ) -> Result<rsmpeg::avutil::AVHWFramesContext, String> {
        let hw_pix_fmt = self.hw_pix_fmt();
        if hw_pix_fmt == ffi::AV_PIX_FMT_NONE {
            return Err(format!(
                "no frame format is known for hardware device `{}` of type {}",
                self.name,
                type_name(self.device_type)
            ));
        }
        // Safety: `as_ptr` is a live device reference, and `av_hwframe_ctx_alloc`
        // takes its own reference to it.
        let raw = std::ptr::NonNull::new(unsafe { ffi::av_hwframe_ctx_alloc(self.as_ptr()) })
            .ok_or_else(|| format!("av_hwframe_ctx_alloc failed for device `{}`", self.name))?;
        // Safety: freshly allocated by the call above, so it is a frames context.
        let mut frames = unsafe { rsmpeg::avutil::AVHWFramesContext::from_raw(raw) };
        {
            let data = frames.data();
            data.format = hw_pix_fmt;
            data.sw_format = sw_pix_fmt;
            data.width = width;
            data.height = height;
        }
        frames
            .init()
            .map_err(|e| format!("av_hwframe_ctx_init failed: {e}"))?;
        Ok(frames)
    }
}

/// The name of a device type, for messages.
pub fn type_name(device_type: ffi::AVHWDeviceType) -> String {
    // Safety: the call returns a static string or null for an unknown type.
    let name = unsafe { ffi::av_hwdevice_get_type_name(device_type) };
    if name.is_null() {
        return "none".into();
    }
    unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

/// Every device this instance has initialized, by name.
pub struct HwAccelService {
    devices: Mutex<HashMap<String, Arc<HwDevice>>>,
}

impl HwAccelService {
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(HashMap::new()),
        }
    }

    /// `hwaccel.init`'s payload: `{"name": …, "type": …, "device": …,
    /// "options": {…}}`. A name that already exists is left alone and reported,
    /// which is C++ `PolicyIfExists::Ignore`.
    pub fn init(&self, params: &Value) -> Result<String, String> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or("hwaccel.init needs a \"name\"")?
            .to_string();
        if name.is_empty() {
            return Err("hwaccel.init needs a non-empty \"name\"".into());
        }
        let type_str = params
            .get("type")
            .and_then(Value::as_str)
            .ok_or("hwaccel.init needs a \"type\", e.g. \"cuda\"")?;
        {
            let devices = self.devices.lock().unwrap();
            if devices.contains_key(&name) {
                log::info!("hwaccel `{name}` already exists, keeping it");
                return Ok(format!("hwaccel `{name}` already exists"));
            }
        }

        let c_type =
            CString::new(type_str).map_err(|_| "the device type contains a NUL".to_string())?;
        let device_type = hwdevice_find_type_by_name(&c_type);
        if device_type == ffi::AV_HWDEVICE_TYPE_NONE {
            return Err(format!(
                "unknown hardware device type `{type_str}`; this build has {}",
                available_types().join(", ")
            ));
        }

        let device_string = match params.get("device") {
            Some(Value::String(s)) => Some(
                CString::new(s.as_str())
                    .map_err(|_| "the device string contains a NUL".to_string())?,
            ),
            Some(Value::Null) | None => None,
            Some(other) => return Err(format!("\"device\" must be a string, got {other}")),
        };
        let options = Options::from_json(params.get("options"))?.into_av_dictionary();

        let ctx =
            AVHWDeviceContext::create(device_type, device_string.as_deref(), options.as_ref(), 0)
                .map_err(|e| {
                format!(
                    "av_hwdevice_ctx_create failed for type `{type_str}`{}: {e}",
                    match &device_string {
                        Some(d) => format!(" device `{}`", d.to_string_lossy()),
                        None => String::new(),
                    }
                )
            })?;
        let device = Arc::new(HwDevice {
            name: name.clone(),
            device_type,
            ctx,
        });
        log::info!(
            "hwaccel `{name}`: opened {type_str} device{}, frames are {}",
            match &device_string {
                Some(d) => format!(" `{}`", d.to_string_lossy()),
                None => String::new(),
            },
            crate::libav::codec::pix_fmt_name(device.hw_pix_fmt())
        );
        self.devices.lock().unwrap().insert(name.clone(), device);
        Ok(format!("hwaccel `{name}` ready"))
    }

    /// The device a node named, or an error naming what does exist. Nodes
    /// resolve at build time, so a typo fails at `node.add` as it does in C++.
    pub fn get(&self, name: &str) -> Result<Arc<HwDevice>, String> {
        let devices = self.devices.lock().unwrap();
        match devices.get(name) {
            Some(device) => Ok(device.clone()),
            None if devices.is_empty() => Err(format!(
                "no hardware device `{name}`: none has been initialized, run `hwaccel.init` first"
            )),
            None => Err(format!(
                "no hardware device `{name}`; initialized: {}",
                devices.keys().cloned().collect::<Vec<_>>().join(", ")
            )),
        }
    }

    pub fn names(&self) -> Vec<String> {
        self.devices.lock().unwrap().keys().cloned().collect()
    }
}

impl Default for HwAccelService {
    fn default() -> Self {
        Self::new()
    }
}

/// Every device type this libavutil was built with.
fn available_types() -> Vec<String> {
    let mut out = Vec::new();
    let mut t = ffi::AV_HWDEVICE_TYPE_NONE;
    loop {
        // Safety: the iterator terminates by returning `NONE`.
        t = unsafe { ffi::av_hwdevice_iterate_types(t) };
        if t == ffi::AV_HWDEVICE_TYPE_NONE {
            break;
        }
        out.push(type_name(t));
    }
    if out.is_empty() {
        out.push("none".into());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_device_needs_a_name_and_a_known_type() {
        let service = HwAccelService::new();
        assert!(
            service
                .init(&json!({"type": "cuda"}))
                .unwrap_err()
                .contains("name")
        );
        assert!(
            service
                .init(&json!({"name": "gpu"}))
                .unwrap_err()
                .contains("type")
        );
        let message = service
            .init(&json!({"name": "gpu", "type": "definitely-not-a-device"}))
            .unwrap_err();
        assert!(
            message.contains("unknown hardware device type"),
            "{message}"
        );
    }

    #[test]
    fn an_unknown_name_says_what_is_available() {
        let service = HwAccelService::new();
        let message = service.get("gpu").unwrap_err();
        assert!(message.contains("none has been initialized"), "{message}");
    }
}
