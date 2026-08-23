export type PixelFormat = 'bgra' | 'rgba';

export interface TexInfo {
  readonly codedWidth: number;
  readonly codedHeight: number;
  readonly stride: number;
  readonly pixelFormat: PixelFormat;
  readonly modifier: bigint;
  readonly offset: bigint;
  readonly timestampNs: bigint;
  readonly frameNumber: bigint;
}

export interface TraceTexInfo extends TexInfo {
  readonly traceId: bigint;
  readonly sequence: bigint;
  readonly sourcePtsMs: bigint;
  readonly rendererReceivedNs: bigint;
  readonly rafNs: bigint;
  readonly domAppliedNs: bigint;
  readonly paintNs: bigint;
  readonly dmabufSendNs: bigint;
}

const PIX_FMT_BGRA = 0x42475241; // "BGRA" little-endian
const PIX_FMT_RGBA = 0x52474241; // "RGBA" little-endian

export const TEX_INFO_HEADER_BYTES = 48;
export const TRACE_TEX_INFO_MAGIC = 0x31544253; // "SBT1" little-endian
export const TRACE_TEX_INFO_VERSION = 1;
export const TRACE_TEX_INFO_HEADER_BYTES = 120;

/**
 * Encodes a 48-byte little-endian header sent ahead of every dmabuf FD.
 * Layout (offsets in bytes):
 *   0:  u32 codedWidth
 *   4:  u32 codedHeight
 *   8:  u32 stride
 *   12: u32 pixelFormat (FOURCC: 'BGRA' or 'RGBA')
 *   16: u64 modifier
 *   24: u64 offset
 *   32: u64 timestampNs   (nanoseconds from CLOCK_MONOTONIC)
 *   40: u64 frameNumber
 *
 * Must remain wire-compatible with the consumer of `/tmp/dma-page/{id}.sock`.
 * Locked by tests/unit/capture/TexInfoEncoder.test.ts.
 */
export class TexInfoEncoder {
  public encode(info: TexInfo): Buffer {
    const buf = Buffer.allocUnsafe(TEX_INFO_HEADER_BYTES);
    buf.writeUInt32LE(info.codedWidth >>> 0, 0);
    buf.writeUInt32LE(info.codedHeight >>> 0, 4);
    buf.writeUInt32LE(info.stride >>> 0, 8);
    buf.writeUInt32LE(this.pixelFormatCode(info.pixelFormat), 12);
    buf.writeBigUInt64LE(info.modifier, 16);
    buf.writeBigUInt64LE(info.offset, 24);
    buf.writeBigUInt64LE(info.timestampNs, 32);
    buf.writeBigUInt64LE(info.frameNumber, 40);
    return buf;
  }

  public encodeTrace(info: TraceTexInfo): Buffer {
    const buf = Buffer.allocUnsafe(TRACE_TEX_INFO_HEADER_BYTES);
    this.encode(info).copy(buf, 0);
    buf.writeUInt32LE(TRACE_TEX_INFO_MAGIC, 48);
    buf.writeUInt16LE(TRACE_TEX_INFO_VERSION, 52);
    buf.writeUInt16LE(TRACE_TEX_INFO_HEADER_BYTES, 54);
    buf.writeBigUInt64LE(info.traceId, 56);
    buf.writeBigUInt64LE(info.sequence, 64);
    buf.writeBigInt64LE(info.sourcePtsMs, 72);
    buf.writeBigUInt64LE(info.rendererReceivedNs, 80);
    buf.writeBigUInt64LE(info.rafNs, 88);
    buf.writeBigUInt64LE(info.domAppliedNs, 96);
    buf.writeBigUInt64LE(info.paintNs, 104);
    buf.writeBigUInt64LE(info.dmabufSendNs, 112);
    return buf;
  }

  private pixelFormatCode(fmt: PixelFormat): number {
    return fmt === 'rgba' ? PIX_FMT_RGBA : PIX_FMT_BGRA;
  }
}
