import { EventEmitter } from 'node:events';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebContents } from 'electron';
import { AllowedDims } from '../../../src/main/capture/AllowedDims';
import { FrameCaptureChannel } from '../../../src/main/capture/FrameCaptureChannel';
import type { LogSink } from '../../../src/main/support/Logger';
import type { SendResult } from 'fdpass';

const fdpass = vi.hoisted(() => ({
  broadcastFd: vi.fn<() => Promise<SendResult>>(),
  closeServer: vi.fn<() => bigint[]>().mockReturnValue([]),
  createServer: vi.fn().mockReturnValue(true),
  monotonicTimeNs: vi.fn().mockReturnValue(93_000_000_000_000n),
  setReleaseCallback: vi.fn(),
  setServerLogger: vi.fn(),
}));

vi.mock('fdpass', () => fdpass);

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
    fdpass.broadcastFd.mockResolvedValue({clients: 1, sent: 1, backpressure: 0, disconnected: 0, errors: 0});
    fdpass.closeServer.mockReturnValue([]);
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
      frameNumber: bigint, reusable: boolean,
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

    releaseFrame(0n, true);

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

  it.each([
    [{clients: 0, sent: 0, backpressure: 0, disconnected: 0, errors: 0}, 0, {no_consumer: 1}],
    [{clients: 2, sent: 1, backpressure: 1, disconnected: 0, errors: 0}, 1, {fdpass_backpressure: 1}],
    [{clients: 2, sent: 0, backpressure: 0, disconnected: 2, errors: 0}, 0, {fdpass_disconnected: 2}],
    [{clients: 1, sent: 0, backpressure: 0, disconnected: 0, errors: 1}, 0, {fdpass_errors: 1}],
  ] as const)('counts actual and partial deliveries: %j', async (result, sent, reasons) => {
    fdpass.broadcastFd.mockResolvedValue(result);
    const channel = new FrameCaptureChannel({socketPath: '/tmp/dma-page/counts.sock',
      allowedDims: AllowedDims.fromList(['1080x1920']), log: logSink()});
    const contents = new EventEmitter() as WebContents & EventEmitter;
    channel.attach(contents);
    await channel.start();
    contents.emit('paint', makeTexture(10), {}, image());
    await Promise.resolve();
    expect(channel.getStats().txFrameCount).toBe(sent);
    expect(channel.getStats().droppedFrames).toBe(1);
    expect(channel.getStats().droppedReasons).toEqual(reasons);
    await channel.stop();
  });

  it.each(['disconnect', 'stop'] as const)('quarantines outstanding textures on %s', async (reason) => {
    const options = {socketPath: `/tmp/dma-page/quarantine-${reason}.sock`,
      allowedDims: AllowedDims.fromList(['1080x1920']), log: logSink()};
    const channel = new FrameCaptureChannel(options);
    const contents = Object.assign(new EventEmitter(), {stopPainting: vi.fn()}) as unknown as WebContents & EventEmitter;
    channel.attach(contents);
    await channel.start();
    const frame = makeTexture(20);
    contents.emit('paint', frame, {}, image());
    await Promise.resolve();
    if (reason === 'disconnect') {
      const callback = fdpass.setReleaseCallback.mock.calls[0]![1] as (n: bigint, reusable: boolean) => void;
      callback(0n, false);
      expect(contents.stopPainting).toHaveBeenCalledTimes(1);
    } else fdpass.closeServer.mockReturnValue([0n]);
    await channel.stop();
    expect(frame.texture.release).not.toHaveBeenCalled();
    expect(channel.getStats().quarantinedFrameCount).toBe(1);
    await expect(new FrameCaptureChannel(options).start()).rejects.toThrow('restart this browser worker');
  });
});
