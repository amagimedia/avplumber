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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodePads {
    pub sources: Vec<PadDecl>,
    pub sinks: Vec<PadDecl>,
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
