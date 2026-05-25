import { describe, expect, it, vi } from 'vitest';
import { WindowManager } from '../../src/main/WindowManager';
import type { IManagedWindow, IManagedWindowFactory } from '../../src/main/ManagedWindow';
import type { WindowConfig, WindowSnapshot } from '../../src/main/config/WindowConfig';
import {
  CapacityError,
  ConflictError,
  NotFoundError,
} from '../../src/main/rest/errors';

function fakeWindow(config: WindowConfig, overrides: Partial<IManagedWindow> = {}): IManagedWindow {
  const snapshot: WindowSnapshot = {
    id: config.id,
    url: config.url,
    width: config.width,
    height: config.height,
    fps: config.fps,
    audio: config.audio,
    visible: false,
    stats: {
      paintCount: 0,
      droppedFrames: 0,
      droppedReasons: {},
      txFrameCount: 0,
      lastPaintTsMs: null,
    },
  };
  return {
    id: config.id,
    create: vi.fn().mockResolvedValue(undefined),
    refresh: vi.fn(),
    update: vi.fn(),
    show: vi.fn(),
    destroy: vi.fn().mockResolvedValue(undefined),
    snapshot: vi.fn().mockReturnValue(snapshot),
    ...overrides,
  };
}

function makeFactory(overrides?: (cfg: WindowConfig) => Partial<IManagedWindow>): {
  factory: IManagedWindowFactory;
  created: IManagedWindow[];
} {
  const created: IManagedWindow[] = [];
  const factory: IManagedWindowFactory = {
    create(cfg: WindowConfig): IManagedWindow {
      const w = fakeWindow(cfg, overrides?.(cfg));
      created.push(w);
      return w;
    },
  };
  return { factory, created };
}

const cfg = (id: string): WindowConfig => ({
  id,
  url: 'https://example.com/',
  width: 1280,
  height: 720,
  fps: 30,
  audio: false,
});

describe('WindowManager.open', () => {
  it('creates and tracks a window', async () => {
    const { factory } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    const snap = await mgr.open(cfg('a'));
    expect(snap.id).toBe('a');
    expect(mgr.status().count).toBe(1);
  });

  it('throws ConflictError on duplicate id', async () => {
    const { factory } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await mgr.open(cfg('a'));
    await expect(mgr.open(cfg('a'))).rejects.toBeInstanceOf(ConflictError);
  });

  it('throws CapacityError at the cap', async () => {
    const { factory } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 2 });
    await mgr.open(cfg('a'));
    await mgr.open(cfg('b'));
    await expect(mgr.open(cfg('c'))).rejects.toBeInstanceOf(CapacityError);
  });

  it('rolls back on create() failure', async () => {
    let calls = 0;
    const { factory, created } = makeFactory(() => {
      const n = calls++;
      if (n === 0) {
        return { create: vi.fn().mockRejectedValue(new Error('boom')) };
      }
      return {};
    });
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await expect(mgr.open(cfg('a'))).rejects.toThrow('boom');
    expect(mgr.status().count).toBe(0);
    expect(created[0]!.destroy).toHaveBeenCalled();
    // After rollback, the same id can be opened again.
    await expect(mgr.open(cfg('a'))).resolves.toBeDefined();
  });
});

describe('WindowManager.close / closeAll', () => {
  it('closes a known id', async () => {
    const { factory, created } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await mgr.open(cfg('a'));
    await mgr.close('a');
    expect(mgr.status().count).toBe(0);
    expect(created[0]!.destroy).toHaveBeenCalled();
  });

  it('NotFoundError for unknown id', async () => {
    const { factory } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await expect(mgr.close('ghost')).rejects.toBeInstanceOf(NotFoundError);
  });

  it('closes all windows', async () => {
    const { factory, created } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await mgr.open(cfg('a'));
    await mgr.open(cfg('b'));
    await mgr.closeAll();
    expect(mgr.status().count).toBe(0);
    expect(created[0]!.destroy).toHaveBeenCalled();
    expect(created[1]!.destroy).toHaveBeenCalled();
  });
});

describe('WindowManager refresh/update/show', () => {
  it('refresh delegates and returns snapshot', async () => {
    const { factory, created } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await mgr.open(cfg('a'));
    expect(mgr.refresh('a').id).toBe('a');
    expect(created[0]!.refresh).toHaveBeenCalled();
  });

  it('refresh throws NotFoundError for unknown id', async () => {
    const { factory } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    expect(() => mgr.refresh('ghost')).toThrow(NotFoundError);
  });

  it('update passes URL through', async () => {
    const { factory, created } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await mgr.open(cfg('a'));
    mgr.update('a', 'https://other.example/');
    expect(created[0]!.update).toHaveBeenCalledWith('https://other.example/');
  });

  it('show passes visibility flag through', async () => {
    const { factory, created } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 8 });
    await mgr.open(cfg('a'));
    mgr.show('a', true);
    expect(created[0]!.show).toHaveBeenCalledWith(true);
  });
});

describe('WindowManager.status', () => {
  it('reports count and max', async () => {
    const { factory } = makeFactory();
    const mgr = new WindowManager(factory, { maxWindows: 4 });
    await mgr.open(cfg('a'));
    const status = mgr.status();
    expect(status.count).toBe(1);
    expect(status.maxWindows).toBe(4);
    expect(status.windows).toHaveLength(1);
  });
});
