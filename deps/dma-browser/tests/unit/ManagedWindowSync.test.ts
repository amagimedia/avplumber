import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { EventEmitter } from 'node:events';
import type { Logger } from '../../src/main/support/Logger';
import { AllowedDims } from '../../src/main/capture/AllowedDims';
import { ManagedWindow } from '../../src/main/ManagedWindow';

interface WindowStub {
  readonly options: { webPreferences: { additionalArguments: string[] } };
  readonly webContents: EventEmitter;
}
const mocks = vi.hoisted(() => ({
  windows: [] as WindowStub[],
  start: vi.fn(),
  stop: vi.fn(),
  realign: vi.fn(),
  alignStagger: vi.fn().mockResolvedValue(true),
}));

vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events');
  return {
    BrowserWindow: class {
      public readonly webContents = Object.assign(new EventEmitter(), {
        setFrameRate: vi.fn(),
        isDestroyed: () => false,
        insertCSS: vi.fn().mockResolvedValue(''),
        executeJavaScript: vi.fn().mockResolvedValue(undefined),
        reloadIgnoringCache: vi.fn(),
      });
      constructor(public readonly options: WindowStub['options']) {
        mocks.windows.push(this);
      }
      public setBounds = vi.fn();
      public loadURL = vi.fn().mockResolvedValue(undefined);
      public isDestroyed = () => false;
      public destroy = vi.fn();
    },
  };
});
vi.mock('../../src/main/capture/FrameCaptureChannel', () => ({
  FrameCaptureChannel: class {
    public start = vi.fn().mockResolvedValue(undefined);
    public stop = vi.fn().mockResolvedValue(undefined);
    public attach = vi.fn();
    public hold = vi.fn();
    public getStats = () => null;
  },
}));
vi.mock('../../src/main/capture/AudioCaptureChannel', () => ({ AudioCaptureChannel: class {} }));
vi.mock('../../src/main/capture/BrowserFrameSync', () => ({
  BrowserFrameSync: class {
    public start = mocks.start;
    public stop = mocks.stop;
    public realign = mocks.realign;
    public snapshot = () => ({ applications: 2 });
  },
}));
vi.mock('../../src/main/capture/FramePhase', () => ({
  alignFramePhase: mocks.alignStagger,
  MONOTONIC_CLOCK: {},
}));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.windows.length = 0;
});

describe('ManagedWindow browser sync', () => {
  it.each([undefined, false, true])(
    'connects lifecycle and preload with loop mode %s',
    async (loopAnimations) => {
      const sync =
        loopAnimations === undefined
          ? {}
          : {
              sync: { epochMs: 1234567890000, phase: 0.25, loopAnimations },
            };
      const window = new ManagedWindow({
        config: {
          id: 'test',
          url: 'https://example.com/',
          width: 640,
          height: 480,
          fps: 30,
          audio: false,
          ...sync,
        },
        socketDir: '/tmp/dma-browser-test',
        preloadPath: '/tmp/dma-browser-test/preload.js',
        logger: {
          forWindow: () => ({ write: vi.fn() }),
          closeWindow: vi.fn(),
        } as unknown as Logger,
        allowedDims: AllowedDims.fromList(['640x480']),
        loadWatchdogMs: 30000,
        retainedFramePoolSize: 4,
        framePhase: 0.5,
      });
      try {
        await window.create();
        const browser = mocks.windows[0]!;
        const args = browser.options.webPreferences.additionalArguments;
        expect(args.includes('--loop-sync-epoch-ms=1234567890000')).toBe(loopAnimations === true);
        browser.webContents.emit('did-navigate');
        browser.webContents.emit('did-finish-load');
        if (loopAnimations === undefined) {
          expect(mocks.start).not.toHaveBeenCalled();
          expect(mocks.alignStagger).toHaveBeenCalledOnce();
          expect(window.snapshot().sync).toBeUndefined();
        } else {
          expect(mocks.start).toHaveBeenCalledOnce();
          expect(mocks.realign).toHaveBeenCalledOnce();
          expect(mocks.alignStagger).not.toHaveBeenCalled();
          expect(window.snapshot().sync).toEqual(sync.sync);
          expect(window.snapshot().syncStatus).toMatchObject({ applications: 2 });
        }
      } finally {
        await window.destroy();
      }
      expect(mocks.stop).toHaveBeenCalledTimes(loopAnimations === undefined ? 0 : 1);
    },
  );
});
