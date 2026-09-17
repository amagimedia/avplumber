//! Owned `AVBSFContext`. rsmpeg wraps a single filter by name; the list
//! syntax (`h264_mp4toannexb`, `dump_extra=freq=keyframe`) needs the C API.

use std::ffi::CString;
use std::ptr::NonNull;

use rsmpeg::avcodec::{AVCodecParameters, AVPacket};
use rusty_ffmpeg::ffi;

use crate::graph::media::AvpRational;
use crate::libav::error::{av_error, is_eagain, is_eof};

/// An `AVBSFContext` produced by `av_bsf_list_parse_str`.
pub struct Context(NonNull<ffi::AVBSFContext>);

// Touched only from the node's thread, under that node's lock.
unsafe impl Send for Context {}

impl Context {
    pub fn parse(list: &str) -> Result<Self, String> {
        let text = CString::new(list).map_err(|_| "bsf: the filter list contains a NUL")?;
        let mut ctx: *mut ffi::AVBSFContext = std::ptr::null_mut();
        let ret = unsafe { ffi::av_bsf_list_parse_str(text.as_ptr(), &mut ctx) };
        if ret < 0 {
            return Err(format!("bsf: cannot create `{list}`: {}", av_error(ret)));
        }
        NonNull::new(ctx)
            .map(Self)
            .ok_or_else(|| format!("bsf: `{list}` produced no context"))
    }

    /// Copies the input parameters in, sets the input time base, initializes.
    pub fn init(
        &mut self,
        par_in: &AVCodecParameters,
        time_base_in: AvpRational,
    ) -> Result<(), String> {
        unsafe {
            let ctx = self.0.as_ptr();
            let ret = ffi::avcodec_parameters_copy((*ctx).par_in, par_in.as_ptr());
            if ret < 0 {
                return Err(format!(
                    "bsf: cannot copy codec parameters: {}",
                    av_error(ret)
                ));
            }
            (*ctx).time_base_in = time_base_in.into();
            let ret = ffi::av_bsf_init(ctx);
            if ret < 0 {
                return Err(format!("bsf: init failed: {}", av_error(ret)));
            }
        }
        Ok(())
    }

    pub fn par_out(&self) -> AVCodecParameters {
        let mut out = AVCodecParameters::new();
        unsafe {
            ffi::avcodec_parameters_copy(out.as_mut_ptr(), (*self.0.as_ptr()).par_out);
        }
        out
    }

    pub fn time_base_out(&self) -> AvpRational {
        unsafe { (*self.0.as_ptr()).time_base_out }.into()
    }

    /// `None` signals the end of the stream. The filter takes the packet.
    fn send(&mut self, packet: Option<&mut AVPacket>) -> Result<(), i32> {
        let ptr = packet.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr());
        let ret = unsafe { ffi::av_bsf_send_packet(self.0.as_ptr(), ptr) };
        if ret >= 0 || is_eof(ret) {
            Ok(())
        } else {
            Err(ret)
        }
    }

    /// The next filtered packet, or `None` when the filter wants input or is
    /// drained.
    fn receive(&mut self) -> Result<Option<AVPacket>, String> {
        let mut out = AVPacket::new();
        let ret = unsafe { ffi::av_bsf_receive_packet(self.0.as_ptr(), out.as_mut_ptr()) };
        if ret >= 0 {
            return Ok(Some(out));
        }
        if is_eagain(ret) || is_eof(ret) {
            return Ok(None);
        }
        Err(format!("bsf: receive failed: {}", av_error(ret)))
    }

    /// Sends one packet (or the end) and returns everything the filter gives
    /// back. `AVERROR(EAGAIN)` on send means receive first, then retry.
    pub fn run(&mut self, mut packet: Option<&mut AVPacket>) -> Result<Vec<AVPacket>, String> {
        let mut out = Vec::new();
        loop {
            match self.send(packet.as_deref_mut()) {
                Ok(()) => break,
                Err(code) if is_eagain(code) => match self.receive()? {
                    Some(pkt) => out.push(pkt),
                    None => {
                        return Err("bsf: send wants output but receive is empty".into());
                    }
                },
                Err(code) => return Err(format!("bsf: send failed: {}", av_error(code))),
            }
        }
        while let Some(pkt) = self.receive()? {
            out.push(pkt);
        }
        Ok(out)
    }

    pub fn flush(&mut self) {
        unsafe { ffi::av_bsf_flush(self.0.as_ptr()) };
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let mut ctx = self.0.as_ptr();
        unsafe { ffi::av_bsf_free(&mut ctx) };
    }
}
