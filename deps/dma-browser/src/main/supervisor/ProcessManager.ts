import type { WindowConfig, WindowSnapshot } from '../config/WindowConfig';
import type { StatusReport, WindowControl } from '../WindowControl';
import { CapacityError, ConflictError, NotFoundError } from '../rest/errors';
import type { BrowserWorker, WorkerStatus } from './WorkerProcess';

export interface MultiprocessStatusReport extends StatusReport {
  readonly workers: readonly WorkerStatus[];
}

export class ProcessManager implements WindowControl {
  private readonly workers: readonly BrowserWorker[];
  private readonly maxWindows: number;
  private readonly owners = new Map<string, BrowserWorker>();

  constructor(
    workers: readonly BrowserWorker[],
    maxWindows = workers.reduce((sum, worker) => sum + worker.maxWindows, 0),
  ) {
    if (workers.length === 0) throw new Error('At least one Electron worker is required');
    const workerCapacity = workers.reduce((sum, worker) => sum + worker.maxWindows, 0);
    if (maxWindows < 1 || maxWindows > workerCapacity) {
      throw new Error('Global window capacity must fit within Electron worker capacity');
    }
    this.workers = workers;
    this.maxWindows = maxWindows;
  }

  public async start(): Promise<void> {
    await Promise.all(this.workers.map(async (worker) => worker.start()));
  }

  public async stop(): Promise<void> {
    await Promise.all(this.workers.map(async (worker) => worker.stop()));
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
    const worker = this.workers
      .filter((candidate) => candidate.hasCapacity)
      .sort((left, right) => left.desiredCount - right.desiredCount || left.index - right.index)[0];
    if (!worker) {
      throw new CapacityError(
        `All ${String(this.workers.length)} Electron workers are at capacity`,
      );
    }
    this.owners.set(config.id, worker);
    try {
      return await worker.open(config);
    } catch (err) {
      this.owners.delete(config.id);
      throw err;
    }
  }

  public async close(id: string): Promise<void> {
    const worker = this.requireOwner(id);
    this.owners.delete(id);
    await worker.close(id);
  }

  public async closeAll(): Promise<void> {
    this.owners.clear();
    await Promise.all(this.workers.map(async (worker) => worker.closeAll()));
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
    const results = await Promise.allSettled(this.workers.map(async (worker) => worker.status()));
    const windows: WindowSnapshot[] = [];
    const workerStatuses: WorkerStatus[] = [];
    for (let index = 0; index < results.length; index += 1) {
      const worker = this.workers[index];
      const result = results[index];
      if (!worker || !result) continue;
      if (result.status === 'fulfilled') {
        windows.push(...result.value.windows);
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

  private requireOwner(id: string): BrowserWorker {
    const worker = this.owners.get(id);
    if (!worker) throw new NotFoundError(`No window with id "${id}"`);
    return worker;
  }
}
