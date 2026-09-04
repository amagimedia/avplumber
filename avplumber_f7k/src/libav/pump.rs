//! The `send` / `receive` / `EAGAIN` protocol decode and encode share.
//!
//! libavcodec's contract is that `send` may refuse an input with `EAGAIN`, which
//! means "take my output first and offer the same input again". A node cannot
//! park inside that retry — it still has to answer its output edge — so the
//! refused input is stashed here and the outputs queue up until the node has
//! pushed them.
//!
//! One struct rather than a trait per direction: both directions move [`Media`]
//! in and [`Media`] out, so the only real difference is which pair of libav
//! calls to make.

use std::collections::VecDeque;

use rsmpeg::avcodec::AVCodecContext;

use crate::graph::buffer::AvpMediaType;
use crate::graph::media::Media;
use crate::libav::error::{av_error, code_of, is_eagain, is_eof};

/// C++ `dec_errors_ > 200`: a few corrupt packets are normal, a stream of them
/// is a real failure.
const ERROR_TOLERANCE: usize = 200;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PumpKind {
    Decode,
    Encode,
}

impl PumpKind {
    fn what(self) -> &'static str {
        match self {
            PumpKind::Decode => "decoder",
            PumpKind::Encode => "encoder",
        }
    }
}

/// Whether one [`Pump::drive`] moved anything. `Stalled` means the codec neither
/// took the stashed input nor produced output, so the caller should wait rather
/// than spin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Progress {
    Moved,
    Stalled,
}

pub struct Pump {
    kind: PumpKind,
    /// What a received frame becomes; ignored when encoding.
    out_media: AvpMediaType,
    node: String,
    /// An input the codec refused with `EAGAIN`, to be offered again.
    stash: Option<Media>,
    ready: VecDeque<Media>,
    errors: usize,
    /// The codec reported `AVERROR_EOF`: it will not produce anything more.
    drained: bool,
}

impl Pump {
    pub fn new(kind: PumpKind, out_media: AvpMediaType, node: &str) -> Self {
        Self {
            kind,
            out_media,
            node: node.into(),
            stash: None,
            ready: VecDeque::new(),
            errors: 0,
            drained: false,
        }
    }

    /// Forget every buffer and every error, for `start` and `FlushStart`. The
    /// caller flushes the codec itself, since it owns the context.
    pub fn reset(&mut self) {
        self.stash = None;
        self.ready.clear();
        self.errors = 0;
        self.drained = false;
    }

    /// Whether the pump is holding an input the codec has not accepted yet.
    pub fn is_loaded(&self) -> bool {
        self.stash.is_some()
    }

    /// Hands the pump one input. Only call with [`Self::is_loaded`] false:
    /// the codec must accept the current input before the next one is read.
    pub fn load(&mut self, input: Media) {
        debug_assert!(self.stash.is_none(), "loaded a pump that still holds input");
        self.stash = Some(input);
    }

    pub fn take_output(&mut self) -> Option<Media> {
        self.ready.pop_front()
    }

    pub fn has_output(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Offers the stashed input (if any), then drains everything the codec will
    /// give. `Err` only for a failure past the error tolerance.
    pub fn drive(&mut self, ctx: &mut AVCodecContext) -> Result<Progress, String> {
        let mut moved = false;
        if let Some(input) = self.stash.take() {
            match self.send(ctx, Some(&input)) {
                Ok(()) => {
                    self.errors = 0;
                    moved = true;
                }
                Err(Refused::Again) => self.stash = Some(input),
                Err(Refused::Flushed) => {
                    log::warn!(
                        "{}: {} is flushed, dropping an input it can no longer take",
                        self.node,
                        self.kind.what()
                    );
                    moved = true;
                }
                Err(Refused::Failed(message)) => {
                    // C++ counts the input as consumed and carries on: one bad
                    // packet should not end a stream.
                    self.note_error(&message)?;
                    moved = true;
                }
            }
        }
        moved |= self.receive_all(ctx)?;
        Ok(if moved {
            Progress::Moved
        } else {
            Progress::Stalled
        })
    }

    /// The end-of-stream sequence: `send(NULL)`, then drain until the codec says
    /// EOF. Errors here are logged and ignored, exactly as C++ does — a flush
    /// failure must not turn a finished stream into a failed one.
    pub fn flush(&mut self, ctx: &mut AVCodecContext) {
        if let Some(pending) = self.stash.take() {
            // Give it one last chance, then let it go: the stream is over.
            if self.send(ctx, Some(&pending)).is_err() {
                log::warn!(
                    "{}: {} never took the last input, dropping it",
                    self.node,
                    self.kind.what()
                );
            }
        }
        if !self.drained {
            match self.send(ctx, None) {
                Ok(()) | Err(Refused::Flushed) => {}
                Err(Refused::Again) => log::warn!(
                    "{}: {} refused the flush request",
                    self.node,
                    self.kind.what()
                ),
                Err(Refused::Failed(message)) => log::warn!(
                    "{}: flushing the {} failed: {message}",
                    self.node,
                    self.kind.what()
                ),
            }
        }
        while !self.drained {
            match self.receive_one(ctx) {
                Ok(Some(out)) => self.ready.push_back(out),
                Ok(None) => break,
                Err(Refused::Flushed) => self.drained = true,
                Err(Refused::Again) => break,
                Err(Refused::Failed(message)) => {
                    log::warn!(
                        "{}: draining the {} failed: {message}",
                        self.node,
                        self.kind.what()
                    );
                    break;
                }
            }
        }
    }

    fn receive_all(&mut self, ctx: &mut AVCodecContext) -> Result<bool, String> {
        let mut got = false;
        loop {
            match self.receive_one(ctx) {
                Ok(Some(out)) => {
                    self.ready.push_back(out);
                    self.errors = 0;
                    got = true;
                }
                Ok(None) | Err(Refused::Again) => return Ok(got),
                Err(Refused::Flushed) => {
                    self.drained = true;
                    return Ok(got);
                }
                Err(Refused::Failed(message)) => {
                    self.note_error(&message)?;
                    return Ok(got);
                }
            }
        }
    }

    fn receive_one(&mut self, ctx: &mut AVCodecContext) -> Result<Option<Media>, Refused> {
        if self.drained {
            return Ok(None);
        }
        match self.kind {
            PumpKind::Decode => {
                let frame = ctx.receive_frame().map_err(refused)?;
                Ok(Some(match self.out_media {
                    AvpMediaType::AUDIO => Media::Audio(frame),
                    _ => Media::Video(frame),
                }))
            }
            PumpKind::Encode => {
                let packet = ctx.receive_packet().map_err(refused)?;
                Ok(Some(Media::Packet(packet)))
            }
        }
    }

    fn send(&self, ctx: &mut AVCodecContext, input: Option<&Media>) -> Result<(), Refused> {
        match self.kind {
            PumpKind::Decode => {
                let packet = match input {
                    None => None,
                    Some(Media::Packet(packet)) => Some(packet),
                    Some(other) => {
                        return Err(Refused::Failed(format!(
                            "expected a packet, got {:?}",
                            other.media_type()
                        )));
                    }
                };
                ctx.send_packet(packet).map_err(refused)
            }
            PumpKind::Encode => {
                let frame = match input {
                    None => None,
                    Some(Media::Video(frame) | Media::Audio(frame)) => Some(frame),
                    Some(other) => {
                        return Err(Refused::Failed(format!(
                            "expected a frame, got {:?}",
                            other.media_type()
                        )));
                    }
                };
                ctx.send_frame(frame).map_err(refused)
            }
        }
    }

    fn note_error(&mut self, message: &str) -> Result<(), String> {
        self.errors += 1;
        if self.errors > ERROR_TOLERANCE {
            return Err(format!(
                "{} failed {} times in a row, last: {message}",
                self.kind.what(),
                self.errors
            ));
        }
        log::warn!("{}: {} error: {message}", self.node, self.kind.what());
        Ok(())
    }
}

/// Why libav would not do what it was asked, split into the two codes the
/// protocol is built on plus everything else.
enum Refused {
    /// `EAGAIN`: correct usage, try the other direction first.
    Again,
    /// `AVERROR_EOF`: fully drained.
    Flushed,
    Failed(String),
}

fn refused(error: rsmpeg::error::RsmpegError) -> Refused {
    match code_of(&error) {
        Some(code) if is_eagain(code) => Refused::Again,
        Some(code) if is_eof(code) => Refused::Flushed,
        Some(code) => Refused::Failed(av_error(code)),
        None => Refused::Failed(error.to_string()),
    }
}
