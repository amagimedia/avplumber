import { describe, expect, it } from 'vitest';
import type { WindowConfig, WindowSnapshot } from '../../../src/main/config/WindowConfig';
import { ProcessManager } from '../../../src/main/supervisor/ProcessManager';
import type { BrowserWorker, WorkerStatus } from '../../../src/main/supervisor/WorkerProcess';
import { CapacityError, ConflictError } from '../../../src/main/rest/errors';
import type { StatusReport } from '../../../src/main/WindowControl';

function snapshot(config: WindowConfig): WindowSnapshot {
  return {
    ...config,
    visible: false,
    stats: {
      paintCount: 0,
      droppedFrames: 0,
      droppedReasons: {},
      txFrameCount: 0,
      releasedFrameCount: 0,
      retainedFrameCount: 0,
      lastPaintTsMs: null,
    },
  };
}

class FakeWorker implements BrowserWorker {
  public readonly windows = new Map<string, WindowSnapshot>();
  public started = false;
  public stopped = false;

  constructor(
    public readonly index: number,
    public readonly port: number,
    public readonly maxWindows: number,
  ) {}

  public get desiredCount(): number {
    return this.windows.size;
  }

  public get hasCapacity(): boolean {
    return this.windows.size < this.maxWindows;
  }

  public async start(): Promise<void> {
    this.started = true;
  }

  public async stop(): Promise<void> {
    this.stopped = true;
  }

  public async open(config: WindowConfig): Promise<WindowSnapshot> {
    const value = snapshot(config);
    this.windows.set(config.id, value);
    return value;
  }

  public async close(id: string): Promise<void> {
    this.windows.delete(id);
  }

  public async closeAll(): Promise<void> {
    this.windows.clear();
  }

  public async refresh(id: string): Promise<WindowSnapshot> {
    return this.require(id);
  }

  public async update(id: string, url: string): Promise<WindowSnapshot> {
    const value = { ...this.require(id), url };
    this.windows.set(id, value);
    return value;
  }

  public async show(id: string, visible: boolean): Promise<WindowSnapshot> {
    const value = { ...this.require(id), visible };
    this.windows.set(id, value);
    return value;
  }

  public async status(): Promise<StatusReport> {
    return {
      windows: Array.from(this.windows.values()),
      count: this.windows.size,
      maxWindows: this.maxWindows,
    };
  }

  public supervisorStatus(error?: string): WorkerStatus {
    return {
      index: this.index,
      port: this.port,
      alive: this.started && !this.stopped,
      ready: this.started && !this.stopped,
      restarts: 0,
      desiredCount: this.windows.size,
      maxWindows: this.maxWindows,
      ...(error === undefined ? {} : { error }),
    };
  }

  private require(id: string): WindowSnapshot {
    const value = this.windows.get(id);
    if (!value) throw new Error(`missing ${id}`);
    return value;
  }
}

function config(index: number): WindowConfig {
  return {
    id: `source_${String(index).padStart(2, '0')}`,
    url: 'https://example.com/',
    width: 1920,
    height: 1080,
    fps: 60,
    audio: false,
  };
}

describe('ProcessManager', () => {
  it('places sixteen windows in sixteen one-window Electron workers', async () => {
    const workers = Array.from(
      { length: 16 },
      (_, index) => new FakeWorker(index, 9010 + index, 1),
    );
    const manager = new ProcessManager(workers);
    await manager.start();
    await Promise.all(Array.from({ length: 16 }, async (_, index) => manager.open(config(index))));

    expect(workers.map((worker) => worker.windows.size)).toEqual(Array(16).fill(1));
    const status = await manager.status();
    expect(status.count).toBe(16);
    expect(status.maxWindows).toBe(16);
    expect(status.workers).toHaveLength(16);
    await expect(manager.open(config(16))).rejects.toBeInstanceOf(CapacityError);
  });

  it('enforces global ids and routes operations to the owning worker', async () => {
    const workers = [new FakeWorker(0, 9010, 1), new FakeWorker(1, 9011, 1)];
    const manager = new ProcessManager(workers);
    await manager.start();
    await manager.open(config(0));
    await expect(manager.open(config(0))).rejects.toBeInstanceOf(ConflictError);

    const updated = await manager.update('source_00', 'https://example.net/');
    expect(updated.url).toBe('https://example.net/');
    expect((await manager.show('source_00', true)).visible).toBe(true);
    await manager.close('source_00');
    expect((await manager.status()).count).toBe(0);
  });

  it('starts, clears, and stops every worker', async () => {
    const workers = [new FakeWorker(0, 9010, 1), new FakeWorker(1, 9011, 1)];
    const manager = new ProcessManager(workers);
    await manager.start();
    await manager.open(config(0));
    await manager.open(config(1));
    await manager.closeAll();
    expect(workers.every((worker) => worker.windows.size === 0)).toBe(true);
    await manager.stop();
    expect(workers.every((worker) => worker.stopped)).toBe(true);
  });

  it('enforces a global capacity below the sum of worker capacities', async () => {
    const workers = Array.from(
      { length: 3 },
      (_, index) => new FakeWorker(index, 9010 + index, 8),
    );
    const manager = new ProcessManager(workers, 20);
    await Promise.all(Array.from({ length: 20 }, async (_, index) => manager.open(config(index))));

    expect(workers.map((worker) => worker.windows.size)).toEqual([7, 7, 6]);
    expect((await manager.status()).maxWindows).toBe(20);
    await expect(manager.open(config(20))).rejects.toBeInstanceOf(CapacityError);
  });
});
