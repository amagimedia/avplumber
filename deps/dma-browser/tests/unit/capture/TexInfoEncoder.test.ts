import { describe, expect, it } from 'vitest';
import {
  TRACE_TEX_INFO_HEADER_BYTES,
  TRACE_TEX_INFO_MAGIC,
  TRACE_TEX_INFO_VERSION,
  TEX_INFO_HEADER_BYTES,
  TexInfoEncoder,
  type TexInfo,
} from '../../../src/main/capture/TexInfoEncoder';

const encoder = new TexInfoEncoder();

const baseInput: TexInfo = {
  codedWidth: 1280,
  codedHeight: 720,
  stride: 1280 * 4,
  pixelFormat: 'bgra',
  modifier: 0xfeedfacedeadbeefn,
  offset: 0x123n,
  timestampNs: 1_700_000_000_000_000_000n,
  frameNumber: 42n,
};

describe('TexInfoEncoder', () => {
  it('produces a 48-byte buffer', () => {
    expect(encoder.encode(baseInput).length).toBe(TEX_INFO_HEADER_BYTES);
  });

  it('locks the exact little-endian byte layout (BGRA)', () => {
    const buf = encoder.encode(baseInput);
    // codedWidth = 1280 = 0x500
    expect(buf.readUInt32LE(0)).toBe(1280);
    expect(buf.readUInt32LE(4)).toBe(720);
    expect(buf.readUInt32LE(8)).toBe(1280 * 4);
    expect(buf.readUInt32LE(12)).toBe(0x42475241); // 'BGRA' LE
    expect(buf.readBigUInt64LE(16)).toBe(0xfeedfacedeadbeefn);
    expect(buf.readBigUInt64LE(24)).toBe(0x123n);
    expect(buf.readBigUInt64LE(32)).toBe(1_700_000_000_000_000_000n);
    expect(buf.readBigUInt64LE(40)).toBe(42n);
  });

  it('encodes RGBA fourcc differently', () => {
    const buf = encoder.encode({ ...baseInput, pixelFormat: 'rgba' });
    expect(buf.readUInt32LE(12)).toBe(0x52474241);
  });

  it('encodes the opt-in trace header without changing the default header', () => {
    const buf = encoder.encodeTrace({
      ...baseInput,
      traceId: 99n,
      sequence: 12n,
      sourcePtsMs: 3456n,
      rendererReceivedNs: 101n,
      rafNs: 102n,
      domAppliedNs: 103n,
      paintNs: 104n,
      dmabufSendNs: 105n,
    });
    expect(buf.length).toBe(TRACE_TEX_INFO_HEADER_BYTES);
    expect(buf.subarray(0, TEX_INFO_HEADER_BYTES)).toEqual(encoder.encode(baseInput));
    expect(buf.readUInt32LE(48)).toBe(TRACE_TEX_INFO_MAGIC);
    expect(buf.readUInt16LE(52)).toBe(TRACE_TEX_INFO_VERSION);
    expect(buf.readUInt16LE(54)).toBe(TRACE_TEX_INFO_HEADER_BYTES);
    expect(buf.readBigUInt64LE(56)).toBe(99n);
    expect(buf.readBigUInt64LE(64)).toBe(12n);
    expect(buf.readBigInt64LE(72)).toBe(3456n);
    expect(buf.readBigUInt64LE(80)).toBe(101n);
    expect(buf.readBigUInt64LE(88)).toBe(102n);
    expect(buf.readBigUInt64LE(96)).toBe(103n);
    expect(buf.readBigUInt64LE(104)).toBe(104n);
    expect(buf.readBigUInt64LE(112)).toBe(105n);
  });

  it('matches a hardcoded hex snapshot to detect drift', () => {
    const buf = encoder.encode({
      codedWidth: 1920,
      codedHeight: 1080,
      stride: 7680,
      pixelFormat: 'bgra',
      modifier: 0x0n,
      offset: 0x0n,
      timestampNs: 0x0n,
      frameNumber: 0x0n,
    });
    // 0x00000780 0x00000438 0x00001e00 0x42475241 then 24 zero bytes
    expect(buf.toString('hex')).toBe(
      [
        '80070000', // 1920 LE
        '38040000', // 1080 LE
        '001e0000', // 7680 LE
        '41524742', // 'BGRA' LE
        '0000000000000000',
        '0000000000000000',
        '0000000000000000',
        '0000000000000000',
      ].join(''),
    );
  });
});
