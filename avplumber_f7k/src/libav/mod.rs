//! libav-facing helpers shared by the media nodes. Node logic lives in
//! `src/nodes/`; everything here is about talking to libav.
//!
//! Buffer-level helpers stay where they already are: `PacketExt`/`FrameExt` in
//! `graph/media.rs`, rational math in `graph/timebase.rs`.

pub mod codec;
pub mod dict;
pub mod error;
pub mod pump;
