import { EventEmitter } from 'node:events';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebContents } from 'electron';
import { AllowedDims } from '../../../src/main/capture/AllowedDims';
import { FrameCaptureChannel } from '../../../src/main/capture/FrameCaptureChannel';
import type { LogSink } from '../../../src/main/support/Logger';

const fdpass = vi.hoisted(() => ({
  broadcastFd: vi
    .fn<(socketPath: string, fd: number, header: Buffer) => Promise<void>>()
    .mockResolvedValue(undefined),
  closeServer: vi.fn(),
  createServer: vi.fn().mockReturnValue(true),
  monotonicTimeNs: vi.fn().mockReturnValue(93_000_000_000_000n),
  setReleaseCallback: vi.fn(),
  setServerLogger: vi.fn(),
}));

const electron = vi.hoisted(() => ({
  ipcMain: { on: vi.fn(), removeListener: vi.fn() },
}));

vi.mock('fdpass', () => fdpass);
vi.mock('electron', () => electron);

function logSink(): LogSink {
  return {
    write: vi.fn(),
    close: vi.fn(),
  };
}

function makeTexture(fd: number): {
  texture: {
    textureInfo: Record<string, unknown>;
    release: ReturnType<typeof vi.fn>;
  };
} {
  return {
    texture: {
      textureInfo: {
        codedSize: { width: 1080, height: 1920 },
        pixelFormat: 'bgra',
        handle: {
          nativePixmap: {
            modifier: 0,
            planes: [{ fd, stride: 1080 * 4, offset: 0 }],
          },
        },
      },
      release: vi.fn(),
    },
  };
}

function image(): { getSize(): { width: number; height: number } } {
  return {
    getSize: () => ({ width: 1080, height: 1920 }),
  };
}

describe('FrameCaptureChannel retained frame lifetime', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fdpass.broadcastFd.mockResolvedValue(undefined);
  });

  it('binds one renderer trace to the next paint and emits the 120-byte header', async () => {
    const channel = new FrameCaptureChannel({
      socketPath: '/tmp/dma-page/scoreboard.sock',
      allowedDims: AllowedDims.fromList(['1080x1920']),
      log: logSink(),
      retainedFramePoolSize: 3,
      windowId: 'scoreboard',
      traceProtocol: true,
    });
    const webContents = Object.assign(new EventEmitter(), { send: vi.fn() }) as WebContents &
      EventEmitter & { send: ReturnType<typeof vi.fn> };
    channel.attach(webContents);
    await channel.start();
    const traceHandler = electron.ipcMain.on.mock.calls[0]![1] as (
      event: { sender: WebContents },
      payload: Record<string, unknown>,
    ) => void;
    traceHandler(
      { sender: webContents },
      {
        windowId: 'scoreboard',
        traceId: 77n,
        sequence: 9n,
        sourcePtsMs: 12_345n,
        rendererReceivedNs: 10n,
        rafNs: 20n,
        domAppliedNs: 30n,
      },
    );

    webContents.emit('paint', makeTexture(8), {}, image());
    await Promise.resolve();

    const header = fdpass.broadcastFd.mock.calls[0]![2];
    expect(header.length).toBe(120);
    expect(header.readBigUInt64LE(56)).toBe(77n);
    expect(header.readBigUInt64LE(64)).toBe(9n);
    expect(header.readBigInt64LE(72)).toBe(12_345n);
    expect(webContents.send).toHaveBeenCalledWith(
      'dma-browser:trace-painted',
      expect.objectContaining({ traceIdStr: '77', sequenceStr: '9' }),
    );
    await channel.stop();
  });

  it('holds transmitted textures until ACK and drops new frames at the in-flight limit', async () => {
    const channel = new FrameCaptureChannel({
      socketPath: '/tmp/dma-page/overlay.sock',
      allowedDims: AllowedDims.fromList(['1080x1920']),
      log: logSink(),
      retainedFramePoolSize: 3,
    });
    const webContents = new EventEmitter() as WebContents & EventEmitter;
    channel.attach(webContents);
    await channel.start();

    const releaseFrame = fdpass.setReleaseCallback.mock.calls[0]![1] as (
      frameNumber: bigint,
    ) => void;

    const frames = [0, 1, 2, 3, 4].map((fd) => makeTexture(fd));
    for (const frame of frames) {
      webContents.emit('paint', frame, {}, image());
    }
    await Promise.resolve();

    expect(fdpass.broadcastFd).toHaveBeenCalledTimes(3);
    expect(fdpass.monotonicTimeNs).toHaveBeenCalledTimes(3);
    expect(channel.getStats().txFrameCount).toBe(3);
    expect(channel.getStats().retainedFrameCount).toBe(3);
    expect(channel.getStats().droppedReasons).toEqual({ retained_pool_full: 2 });
    expect(frames[0]!.texture.release).not.toHaveBeenCalled();
    expect(frames[1]!.texture.release).not.toHaveBeenCalled();
    expect(frames[2]!.texture.release).not.toHaveBeenCalled();
    expect(frames[3]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[4]!.texture.release).toHaveBeenCalledTimes(1);

    releaseFrame(0n);

    expect(frames[0]!.texture.release).toHaveBeenCalledTimes(1);
    expect(channel.getStats().releasedFrameCount).toBe(1);
    expect(channel.getStats().retainedFrameCount).toBe(2);

    const resumed = makeTexture(5);
    webContents.emit('paint', resumed, {}, image());
    await Promise.resolve();

    expect(fdpass.broadcastFd).toHaveBeenCalledTimes(4);
    expect(channel.getStats().retainedFrameCount).toBe(3);
    expect(resumed.texture.release).not.toHaveBeenCalled();

    await channel.stop();

    expect(frames[0]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[1]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[2]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[3]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[4]!.texture.release).toHaveBeenCalledTimes(1);
    expect(resumed.texture.release).toHaveBeenCalledTimes(1);
    expect(channel.getStats().releasedFrameCount).toBe(4);
    expect(channel.getStats().retainedFrameCount).toBe(0);
  });
});
