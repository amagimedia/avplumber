//! Pad declarations and media-type checks at connect.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::graph::buffer::AvpMediaType;
use crate::graph::edge::Edge;

pub struct In<T> {
    pub edge: Arc<dyn Edge>,
    _t: PhantomData<T>,
}

pub struct Out<T> {
    pub edge: Arc<dyn Edge>,
    _t: PhantomData<T>,
}

impl<T> In<T> {
    pub fn new(edge: Arc<dyn Edge>) -> Self {
        Self {
            edge,
            _t: PhantomData,
        }
    }
}
impl<T> Out<T> {
    pub fn new(edge: Arc<dyn Edge>) -> Self {
        Self {
            edge,
            _t: PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PadDecl {
    pub name: String,
    pub media: AvpMediaType,
}

impl PadDecl {
    pub fn new(name: impl Into<String>, media: AvpMediaType) -> Self {
        Self {
            name: name.into(),
            media,
        }
    }
}

/// The pads a node declares: `sources` are its inputs, `sinks` its outputs.
///
/// The constructors cover the common shapes with the conventional pad names
/// `in` and `out`. A script's `src`/`dst` bind to a lone declared pad whatever
/// it is called, so a node with one pad per side never has to name them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodePads {
    pub sources: Vec<PadDecl>,
    pub sinks: Vec<PadDecl>,
}

impl NodePads {
    /// One input pad `in`, no output: a sink.
    pub fn input(media: AvpMediaType) -> Self {
        Self {
            sources: vec![PadDecl::new("in", media)],
            sinks: Vec::new(),
        }
    }

    /// One output pad `out`, no input: a source.
    pub fn output(media: AvpMediaType) -> Self {
        Self {
            sources: Vec::new(),
            sinks: vec![PadDecl::new("out", media)],
        }
    }

    /// `in` and `out`: a transform.
    pub fn siso(input: AvpMediaType, output: AvpMediaType) -> Self {
        Self {
            sources: vec![PadDecl::new("in", input)],
            sinks: vec![PadDecl::new("out", output)],
        }
    }
}

pub fn check_pad_media(
    producer: AvpMediaType,
    consumer: AvpMediaType,
    prod_pad: &str,
    cons_pad: &str,
) -> Result<(), String> {
    if producer != consumer {
        Err(format!(
            "media type mismatch connecting {prod_pad} ({producer:?}) -> {cons_pad} ({consumer:?})"
        ))
    } else {
        Ok(())
    }
}
