//! libav return codes as messages, and the two codes the pumps branch on.

use rusty_ffmpeg::ffi;

/// `av_err2str`: a human-readable form of a negative libav return code.
pub fn av_error(code: i32) -> String {
    rsmpeg::avutil::err2str(code).unwrap_or_else(|| format!("unknown libav error {code}"))
}

/// `AVERROR(EAGAIN)` — "call me again", not a failure.
pub fn is_eagain(code: i32) -> bool {
    code == ffi::AVERROR(ffi::EAGAIN)
}

/// `AVERROR_EOF` — the stream, decoder or encoder is fully drained.
pub fn is_eof(code: i32) -> bool {
    code == ffi::AVERROR_EOF
}

/// The libav code behind an [`rsmpeg::error::RsmpegError`], for the branches
/// that must tell `EAGAIN`/`EOF` from a real failure.
pub fn code_of(error: &rsmpeg::error::RsmpegError) -> Option<i32> {
    error.raw_error()
}
