#pragma once
// Which composite kernel entry a rect table takes (cuda_rect_scale.cu): the host picks it per frame
// from what the table holds. No CUDA here, so the choice is testable on its own.

namespace avp::mixer {

enum class CompositeEntry {
    Packed4, Yuv, YuvArray, Full, Array, Opacity, OpacityArray, YuvFilter, YuvArrayFilter, Filter, YuvAffine, Affine
};

/// What a table holds: packed RGB(A) layers, faded keys, CUarray sources, filtered or demoted
/// layers, the 4x4 cubic or 8 samples, transformed (sub-pixel or rotated) layers.
struct TableKinds {
    bool rgb = false, fade = false, array = false, filter = false, wide = false, affine = false;
};

/// The entry for a table on a canvas of `planes` planes. A table without a transformed layer takes
/// the entry it always took; the affine entries draw every kind, the lean one YUV from linear buffers.
inline CompositeEntry compositeEntry(const TableKinds &k, int planes) {
    if (planes == 1) return CompositeEntry::Packed4;
    if (k.affine) return !k.rgb && !k.array && !k.fade ? CompositeEntry::YuvAffine : CompositeEntry::Affine;
    if (k.filter)
        return k.rgb || k.wide ? CompositeEntry::Filter : k.array ? CompositeEntry::YuvArrayFilter : CompositeEntry::YuvFilter;
    if (k.fade) return k.array ? CompositeEntry::OpacityArray : CompositeEntry::Opacity;
    if (k.rgb) return k.array ? CompositeEntry::Array : CompositeEntry::Full;
    return k.array ? CompositeEntry::YuvArray : CompositeEntry::Yuv;
}

}
