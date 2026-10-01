import type { WindowConfig, WindowSnapshot } from '../config/WindowConfig';
import type { StatusReport, WindowControl } from '../WindowControl';
import { CapacityError, ConflictError, NotFoundError } from '../rest/errors';
import { settleAll } from '../support/concurrency';
import type { BrowserWorker, WorkerStatus } from './WorkerProcess';

export interface MultiprocessStatusReport extends StatusReport {
  readonly workers: readonly WorkerStatus[];
}

// Electron workers run only while they host windows: a worker starts with its first window
// and stops idleStopMs after its last one closes, so a show without browser sources or key
// pages runs no Electron worker at all. A show's window count (plan) spreads its windows
// evenly over the fewest workers that hold them: 20 windows of 8 per worker run 7/7/6.
export class ProcessManager implements WindowControl {
  private readonly workers: readonly BrowserWorker[];
  private readonly maxWindows: number;
  private readonly idleStopMs: number;
  private readonly owners = new Map<string, BrowserWorker>();
  private readonly idleTimers = new Map<BrowserWorker, NodeJS.Timeout>();
  private plannedWindows = 0;

  constructor(
    workers: readonly BrowserWorker[],
    maxWindows = workers.reduce((sum, worker) => sum + worker.maxWindows, 0),
    idleStopMs = 5_000,
  ) {
    if (workers.length === 0) throw new Error('At least one Electron worker is required');
    const workerCapacity = workers.reduce((sum, worker) => sum + worker.maxWindows, 0);
    if (maxWindows < 1 || maxWindows > workerCapacity) {
      throw new Error('Global window capacity must fit within Electron worker capacity');
    }
    this.workers = workers;
    this.maxWindows = maxWindows;
    this.idleStopMs = idleStopMs;
  }

  public async stop(): Promise<void> {
    for (const timer of this.idleTimers.values()) clearTimeout(timer);
    this.idleTimers.clear();
    await Promise.all(this.workers.map(async (worker) => worker.stop()));
  }

  public plan(windows: number): void {
    this.plannedWindows = Math.min(windows, this.maxWindows);
  }

  public async open(config: WindowConfig): Promise<WindowSnapshot> {
    if (this.owners.has(config.id)) {
      throw new ConflictError(`Window with id "${config.id}" is already open`);
    }
    if (this.owners.size >= this.maxWindows) {
      throw new CapacityError(
        `All ${String(this.maxWindows)} configured browser window slots are in use`,
      );
    }
    // The least-loaded of the fewest workers the planned (or so far opened) windows need.
    const perWorker = this.workers[0]?.maxWindows ?? 1;
    const needed = Math.ceil(Math.max(this.plannedWindows, this.owners.size + 1) / perWorker);
    const worker =
      this.workers
        .slice(0, needed)
        .filter((candidate) => candidate.hasCapacity)
        .sort((left, right) => left.desiredCount - right.desiredCount || left.index - right.index)[0] ??
      this.workers.find((candidate) => candidate.hasCapacity);
    if (!worker) {
      throw new CapacityError(
        `All ${String(this.workers.length)} Electron workers are at capacity`,
      );
    }
    this.owners.set(config.id, worker);
    this.cancelIdleStop(worker);
    try {
      return await worker.open(config);
    } catch (err) {
      this.owners.delete(config.id);
      this.scheduleIdleStop(worker);
      throw err;
    }
  }

  public async close(id: string): Promise<void> {
    const worker = this.requireOwner(id);
    this.owners.delete(id);
    await worker.close(id);
    this.scheduleIdleStop(worker);
  }

  public async closeAll(): Promise<void> {
    this.owners.clear();
    await Promise.all(this.workers.map(async (worker) => worker.closeAll()));
    for (const worker of this.workers) this.scheduleIdleStop(worker);
  }

  public async refresh(id: string): Promise<WindowSnapshot> {
    return this.requireOwner(id).refresh(id);
  }

  public async update(id: string, url: string): Promise<WindowSnapshot> {
    return this.requireOwner(id).update(id, url);
  }

  public async show(id: string, visible: boolean): Promise<WindowSnapshot> {
    return this.requireOwner(id).show(id, visible);
  }

  public async status(): Promise<MultiprocessStatusReport> {
    // An idle, stopped worker has nothing to report and is not an error.
    const results = await Promise.allSettled(
      this.workers.map(async (worker) =>
        worker.desiredCount > 0 || worker.supervisorStatus().alive ? worker.status() : null,
      ),
    );
    const windows: WindowSnapshot[] = [];
    const workerStatuses: WorkerStatus[] = [];
    for (let index = 0; index < results.length; index += 1) {
      const worker = this.workers[index];
      const result = results[index];
      if (!worker || !result) continue;
      if (result.status === 'fulfilled') {
        if (result.value) windows.push(...result.value.windows);
        workerStatuses.push(worker.supervisorStatus());
      } else {
        const error =
          result.reason instanceof Error ? result.reason.message : String(result.reason);
        workerStatuses.push(worker.supervisorStatus(error));
      }
    }
    return {
      windows,
      count: windows.length,
      maxWindows: this.maxWindows,
      workers: workerStatuses,
    };
  }

  public async recover(ids: readonly string[]): Promise<void> {
    const owned = new Set(ids);
    const affected = new Set(ids.map((id) => this.owners.get(id)).filter((w) => w !== undefined));
    const restart: BrowserWorker[] = [];
    for (const worker of affected) {
      const { windows } = await worker.status();
      if (!windows.some((w) => owned.has(w.id) && w.stats.quarantinedFrameCount > 0)) continue;
      if (windows.some((w) => !owned.has(w.id))) {
        throw new ConflictError('Cannot recover a browser worker shared with another consumer');
      }
      restart.push(worker);
    }
    // Each restart boots Electron and reloads its pages; one after another took 40-75 s.
    await settleAll(restart.map(async (worker) => worker.restart()));
  }

  // A setup that closes and reopens its pages keeps the workers warm for idleStopMs.
  private scheduleIdleStop(worker: BrowserWorker): void {
    if (worker.desiredCount > 0 || this.idleTimers.has(worker)) return;
    const timer = setTimeout(() => {
      this.idleTimers.delete(worker);
      worker.stopIfIdle().catch((err: unknown) => {
        console.error(`dma-browser worker ${String(worker.index)} idle stop failed: ${String(err)}`);
      });
    }, this.idleStopMs);
    timer.unref();
    this.idleTimers.set(worker, timer);
  }

  private cancelIdleStop(worker: BrowserWorker): void {
    const timer = this.idleTimers.get(worker);
    if (timer) clearTimeout(timer);
    this.idleTimers.delete(worker);
  }

  private requireOwner(id: string): BrowserWorker {
    const worker = this.owners.get(id);
    if (!worker) throw new NotFoundError(`No window with id "${id}"`);
    return worker;
  }
}
