//! Freeze frame and slate. The shift step never builds these; the node does,
//! after `CorrectionGroup::conceal` has named the slots.

use std::collections::HashMap;
use std::sync::Mutex;

use avplumber_f7k::graph::grain::Grain;

static PICTURES: Mutex<Option<HashMap<String, Grain>>> = Mutex::new(None);

fn pictures() -> std::sync::MutexGuard<'static, Option<HashMap<String, Grain>>> {
    PICTURES.lock().unwrap()
}

/// Keep `frame` under `name` for `backup_picture_buffer` / `initial_picture_buffer`.
pub fn store_picture(name: &str, frame: Grain) {
    pictures()
        .get_or_insert_with(HashMap::new)
        .insert(name.to_string(), frame);
}

pub fn load_picture(name: &str) -> Option<Grain> {
    pictures().as_ref().and_then(|map| map.get(name).cloned())
}

/// One video frame from an image file. The slate is cloned per emitted slot.
#[cfg(feature = "ffmpeg")]
pub fn load_image(path: &str) -> Result<Grain, String> {
    use std::ffi::CString;

    use rsmpeg::avcodec::AVCodecContext;
    use rsmpeg::avformat::AVFormatContextInput;
    use rusty_ffmpeg::ffi;

    let c_path = CString::new(path).map_err(|e| e.to_string())?;
    let mut input = AVFormatContextInput::open(&c_path).map_err(|e| e.to_string())?;
    let (index, codec) = input
        .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no video stream in {path}"))?;
    let mut ctx = AVCodecContext::new(&codec);
    let par_ptr = input.streams()[index].codecpar().as_ptr();
    unsafe {
        let ret = ffi::avcodec_parameters_to_context(ctx.as_mut_ptr(), par_ptr);
        if ret < 0 {
            return Err(format!("backup image {path}: codec parameters ({ret})"));
        }
    }
    ctx.open(None).map_err(|e| e.to_string())?;
    loop {
        let Some(packet) = input.read_packet().map_err(|e| e.to_string())? else {
            break;
        };
        if packet.stream_index != index as i32 {
            continue;
        }
        ctx.send_packet(Some(&packet)).map_err(|e| e.to_string())?;
        if let Ok(frame) = ctx.receive_frame() {
            return Ok(Grain::Video(frame));
        }
    }
    let _ = ctx.send_packet(None);
    ctx.receive_frame()
        .map(Grain::Video)
        .map_err(|_| format!("could not decode a frame from {path}"))
}

#[cfg(not(feature = "ffmpeg"))]
pub fn load_image(path: &str) -> Result<Grain, String> {
    Err(format!(
        "backup_image `{path}` needs the ffmpeg feature; use backup_picture_buffer"
    ))
}
