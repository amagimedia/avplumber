import { EventEmitter } from 'node:events';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebContents } from 'electron';
import { AllowedDims } from '../../../src/main/capture/AllowedDims';
import { FrameCaptureChannel } from '../../../src/main/capture/FrameCaptureChannel';
import type { LogSink } from '../../../src/main/support/Logger';

const fdpass = vi.hoisted(() => ({
  broadcastFd: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
  closeServer: vi.fn(),
  createServer: vi.fn().mockReturnValue(true),
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

describe('FrameCaptureChannel retained frame pool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fdpass.broadcastFd.mockResolvedValue(undefined);
  });

  it('holds the last transmitted dmabuf textures and evicts older frames', async () => {
    const channel = new FrameCaptureChannel({
      socketPath: '/tmp/dma-page/overlay.sock',
      allowedDims: AllowedDims.fromList(['1080x1920']),
      log: logSink(),
      retainedFramePoolSize: 3,
    });
    const webContents = new EventEmitter() as WebContents & EventEmitter;
    channel.attach(webContents);

    const frames = [0, 1, 2, 3, 4].map((fd) => makeTexture(fd));
    for (const frame of frames) {
      webContents.emit('paint', frame, {}, image());
    }
    await Promise.resolve();

    expect(fdpass.broadcastFd).toHaveBeenCalledTimes(5);
    expect(channel.getStats().txFrameCount).toBe(5);
    expect(frames[0]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[1]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[2]!.texture.release).not.toHaveBeenCalled();
    expect(frames[3]!.texture.release).not.toHaveBeenCalled();
    expect(frames[4]!.texture.release).not.toHaveBeenCalled();

    await channel.stop();

    expect(frames[2]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[3]!.texture.release).toHaveBeenCalledTimes(1);
    expect(frames[4]!.texture.release).toHaveBeenCalledTimes(1);
  });
});
