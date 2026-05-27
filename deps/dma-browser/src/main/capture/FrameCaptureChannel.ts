import * as fs from 'node:fs';
import * as fdpass from 'fdpass';
import type { WebContents } from 'electron';
import type { LogSink } from '../support/Logger';
import type { AllowedDims } from './AllowedDims';
import type { CaptureStats, ICaptureChannel } from './ICaptureChannel';
import { TexInfoEncoder, type PixelFormat } from './TexInfoEncoder';

interface PaintEventLike {
  texture?: PaintTextureLike | null;
}

interface PaintTextureLike {
  textureInfo?: Record<string, unknown>;
  toJSON?(): unknown;
  release?(): void;
}

interface PaintImageLike {
  getSize?(): { width: number; height: number };
}

interface TextureMeta {
  fd: number | null;
  stride: number | null;
  offsetBig: bigint;
  modifierBig: bigint;
}

interface SizeReport {
  readonly label: string;
  readonly w: number;
  readonly h: number;
}

interface RetainedFrame {
  readonly texture: PaintTextureLike;
  released: boolean;
}

const SOCKET_MONITOR_INTERVAL_MS = 1000;
const DEFAULT_RETAINED_FRAME_POOL_SIZE = 10;

export interface FrameCaptureChannelOptions {
  readonly socketPath: string;
  readonly allowedDims: AllowedDims;
  readonly log: LogSink;
  readonly paintDiag?: boolean;
  readonly retainedFramePoolSize?: number;
}

/**
 * Subscribes to a WebContents `paint` event when `offscreen.useSharedTexture`
 * is enabled, extracts the dmabuf FD, validates dimensions, encodes a 48-byte
 * TexInfo header, and broadcasts to all clients connected to the per-window
 * Unix socket at `socketPath`.
 */
export class FrameCaptureChannel implements ICaptureChannel {
  private readonly opts: FrameCaptureChannelOptions;
  private readonly encoder = new TexInfoEncoder();
  private readonly stats: MutableStats = {
    paintCount: 0,
    droppedFrames: 0,
    droppedReasons: {},
    txFrameCount: 0,
    lastPaintTsMs: null,
  };

  private socketMonitor: NodeJS.Timeout | null = null;
  private boundContents: WebContents | null = null;
  private readonly retainedFramePoolSize: number;
  private readonly retainedFrames: RetainedFrame[] = [];
  private nextFrameNumber = 0n;
  private readonly boundHandler = (event: unknown, _dirty: unknown, img: unknown) =>
    this.handlePaint(event as PaintEventLike, img as PaintImageLike);

  constructor(opts: FrameCaptureChannelOptions) {
    this.opts = opts;
    this.retainedFramePoolSize = Math.max(
      1,
      Math.floor(opts.retainedFramePoolSize ?? DEFAULT_RETAINED_FRAME_POOL_SIZE),
    );
  }

  public start(): Promise<void> {
    fdpass.setServerLogger(this.opts.socketPath, (line) => this.opts.log.write(line));
    const ok = fdpass.createServer(this.opts.socketPath);
    this.opts.log.write(
      `fdpass.createServer(${this.opts.socketPath}) => ${String(
        ok,
      )}; retainedFramePoolSize=${String(this.retainedFramePoolSize)}`,
    );
    this.socketMonitor = setInterval(() => {
      try {
        if (!fs.existsSync(this.opts.socketPath)) {
          fdpass.setServerLogger(this.opts.socketPath, (line) => this.opts.log.write(line));
          const recreated = fdpass.createServer(this.opts.socketPath);
          this.opts.log.write(`fdpass.createServer resurrect => ${String(recreated)}`);
        }
      } catch (err) {
        this.opts.log.write(`socket monitor error: ${String(err)}`);
      }
    }, SOCKET_MONITOR_INTERVAL_MS);
    return Promise.resolve();
  }

  public attach(webContents: WebContents): void {
    this.boundContents = webContents;
    webContents.on('paint', this.boundHandler);
  }

  public stop(): Promise<void> {
    if (this.socketMonitor) {
      clearInterval(this.socketMonitor);
      this.socketMonitor = null;
    }
    if (this.boundContents) {
      try {
        this.boundContents.removeListener('paint', this.boundHandler);
      } catch {
        // ignore
      }
      this.boundContents = null;
    }
    try {
      fdpass.closeServer(this.opts.socketPath);
    } catch (err) {
      this.opts.log.write(`fdpass.closeServer failed: ${String(err)}`);
    }
    this.releaseAllRetainedFrames();
    return Promise.resolve();
  }

  public getStats(): CaptureStats {
    return {
      paintCount: this.stats.paintCount,
      droppedFrames: this.stats.droppedFrames,
      droppedReasons: { ...this.stats.droppedReasons },
      txFrameCount: this.stats.txFrameCount,
      lastPaintTsMs: this.stats.lastPaintTsMs,
    };
  }

  private handlePaint(event: PaintEventLike, img: PaintImageLike): void {
    this.stats.paintCount++;
    this.stats.lastPaintTsMs = Date.now();
    let tex: PaintTextureLike | null | undefined;
    let retained: RetainedFrame | null = null;
    try {
      tex = event.texture;
      if (!tex) {
        this.dropFrame('no_texture');
        return;
      }
      try {
        // Electron 31+ exposes textureInfo directly on OffscreenSharedTexture.
        // Older builds returned it via toJSON(); keep that as a fallback.
        let info: Record<string, unknown> | undefined = tex.textureInfo;
        if (!info && typeof tex.toJSON === 'function') {
          const js = tex.toJSON() as { textureInfo?: Record<string, unknown> };
          info = js.textureInfo;
        }
        if (!info) {
          if (this.opts.paintDiag && this.stats.paintCount === 1) {
            const keys = Object.keys(tex as object).join(',');
            this.opts.log.write(
              `paint texture has no textureInfo; tex keys=[${keys}] toJSON=${JSON.stringify(
                typeof tex.toJSON === 'function' ? tex.toJSON() : null,
              ).slice(0, 200)}`,
            );
          }
          this.dropFrame('no_texture_info');
          return;
        }
        if (this.opts.paintDiag && this.stats.paintCount === 1) {
          this.opts.log.write(
            `first paint textureInfo keys=[${Object.keys(info).join(',')}] sample=${JSON.stringify(
              info,
              (_k: string, v: unknown): unknown => (typeof v === 'bigint' ? v.toString() : v),
            ).slice(0, 400)}`,
          );
        }
        const meta = this.normalizeTextureInfo(info);
        if (meta.fd === null) {
          this.dropFrame('no_fd');
          return;
        }

        const coded = info.codedSize as { width?: number; height?: number } | undefined;
        const codedW = (coded?.width ?? 0) >>> 0;
        const codedH = (coded?.height ?? 0) >>> 0;
        if (!codedW || !codedH) {
          this.dropFrame('bad_coded_size');
          return;
        }

        const allSizes = this.collectSizes(info, img, codedW, codedH);
        for (const s of allSizes) {
          if (!this.opts.allowedDims.allows(s.w, s.h)) {
            this.dropFrame(`size_${s.label}`);
            return;
          }
        }
        if (allSizes.length >= 2) {
          const base = allSizes[0]!;
          for (let i = 1; i < allSizes.length; i++) {
            const s = allSizes[i]!;
            if (s.w !== base.w || s.h !== base.h) {
              this.dropFrame('size_mismatch');
              return;
            }
          }
        }

        const minStride = (codedW * 4) >>> 0;
        const stride = meta.stride && meta.stride >= minStride ? meta.stride : minStride;
        const pixelFormat = this.pixelFormatOf(info);

        const header = this.encoder.encode({
          codedWidth: codedW,
          codedHeight: codedH,
          stride,
          pixelFormat,
          modifier: meta.modifierBig,
          offset: meta.offsetBig,
          timestampNs: fdpass.monotonicTimeNs(),
          frameNumber: this.nextFrameNumber++,
        });

        retained = this.retainTexture(tex);
        fdpass.broadcastFd(this.opts.socketPath, meta.fd, header).then(
          () => {
            this.stats.txFrameCount++;
          },
          (err: unknown) => {
            this.dropFrame('fdpass_error');
            this.opts.log.write(`broadcastFd failed: ${String(err)}`);
            if (retained) {
              this.releaseRetainedFrame(retained);
            }
          },
        );
      } finally {
        if (!retained) {
          this.releaseTexture(tex);
        }
      }
    } catch (err) {
      this.dropFrame('exception');
      this.opts.log.write(`paint handler exception: ${String(err)}`);
      if (retained) {
        this.releaseRetainedFrame(retained);
      } else if (tex) {
        this.releaseTexture(tex);
      }
    }
  }

  private normalizeTextureInfo(info: Record<string, unknown>): TextureMeta {
    // Electron exposes the dmabuf info in one of two shapes depending on
    // version: a legacy `info.planes[0]` array, or a newer
    // `info.handle.nativePixmap.planes[0]` nested object. The modifier and
    // FD live alongside the planes in each case.
    const handle = (info.handle as Record<string, unknown> | undefined) ?? {};
    const nativePixmap = (handle.nativePixmap as Record<string, unknown> | undefined) ?? {};
    const legacyPlanes = Array.isArray(info.planes)
      ? (info.planes as Record<string, unknown>[])
      : null;
    const npPlanes = Array.isArray(nativePixmap.planes)
      ? (nativePixmap.planes as Record<string, unknown>[])
      : null;
    const planes = legacyPlanes ?? npPlanes ?? [];
    const firstPlane: Record<string, unknown> = planes[0] ?? {};
    const fdRaw = firstPlane.fd;
    const strideRaw = firstPlane.stride;
    const offsetRaw = firstPlane.offset;
    const rawModifier = legacyPlanes
      ? info.modifier
      : (nativePixmap.modifier ?? info.modifier);

    const fd = typeof fdRaw === 'number' && Number.isFinite(fdRaw) ? fdRaw : null;
    const stride =
      typeof strideRaw === 'number' && Number.isFinite(strideRaw) ? strideRaw : null;
    return {
      fd,
      stride,
      offsetBig: this.toUint64BigInt(offsetRaw),
      modifierBig: this.toUint64BigInt(rawModifier),
    };
  }

  private toUint64BigInt(value: unknown): bigint {
    if (typeof value === 'bigint') return value;
    if (typeof value === 'string') {
      try {
        return BigInt(value);
      } catch {
        return 0n;
      }
    }
    if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) {
      return BigInt(value);
    }
    return 0n;
  }

  private pixelFormatOf(info: Record<string, unknown>): PixelFormat {
    const pf = String(info.pixelFormat ?? '').toLowerCase();
    return pf === 'rgba' ? 'rgba' : 'bgra';
  }

  private collectSizes(
    info: Record<string, unknown>,
    img: PaintImageLike,
    codedW: number,
    codedH: number,
  ): SizeReport[] {
    const out: SizeReport[] = [{ label: 'coded', w: codedW, h: codedH }];
    const push = (label: string, obj: unknown): void => {
      if (typeof obj !== 'object' || obj === null) return;
      const o = obj as { width?: number; height?: number };
      const w = (o.width ?? 0) >>> 0;
      const h = (o.height ?? 0) >>> 0;
      if (w > 0 && h > 0) out.push({ label, w, h });
    };
    push('visible', info.visibleSize);
    push('content', info.contentSize);
    push('natural', info.naturalSize);
    push('size', info.size);
    push('display', info.displaySize);
    try {
      const sz = typeof img.getSize === 'function' ? img.getSize() : undefined;
      if (sz?.width && sz.height) out.push({ label: 'img', w: sz.width, h: sz.height });
    } catch {
      // ignore
    }
    return out;
  }

  private dropFrame(reason: string): void {
    this.stats.droppedFrames++;
    this.stats.droppedReasons[reason] = (this.stats.droppedReasons[reason] ?? 0) + 1;
  }

  private releaseTexture(tex: PaintTextureLike): void {
    try {
      tex.release?.();
    } catch (err) {
      this.opts.log.write(`texture release failed: ${String(err)}`);
    }
  }

  private retainTexture(tex: PaintTextureLike): RetainedFrame {
    const retained: RetainedFrame = {
      texture: tex,
      released: false,
    };
    this.retainedFrames.push(retained);
    while (this.retainedFrames.length > this.retainedFramePoolSize) {
      const old = this.retainedFrames.shift();
      if (old) {
        this.releaseRetainedFrame(old);
      }
    }
    return retained;
  }

  private releaseRetainedFrame(retained: RetainedFrame): void {
    if (retained.released) return;
    retained.released = true;
    const idx = this.retainedFrames.indexOf(retained);
    if (idx >= 0) {
      this.retainedFrames.splice(idx, 1);
    }
    this.releaseTexture(retained.texture);
  }

  private releaseAllRetainedFrames(): void {
    for (const retained of [...this.retainedFrames]) {
      this.releaseRetainedFrame(retained);
    }
  }
}

interface MutableStats {
  paintCount: number;
  droppedFrames: number;
  droppedReasons: Record<string, number>;
  txFrameCount: number;
  lastPaintTsMs: number | null;
}
