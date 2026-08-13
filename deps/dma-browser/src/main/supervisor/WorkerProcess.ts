import { spawn, type ChildProcess } from 'node:child_process';
import * as http from 'node:http';
import * as path from 'node:path';
import type { WindowConfig, WindowSnapshot } from '../config/WindowConfig';
import type { StatusReport } from '../WindowControl';
import {
  CapacityError,
  ConflictError,
  NotFoundError,
  ValidationError,
  WorkerUnavailableError,
} from '../rest/errors';

interface DesiredWindow {
  readonly config: WindowConfig;
  readonly visible: boolean;
}

export interface WorkerStatus {
  readonly index: number;
  readonly port: number;
  readonly alive: boolean;
  readonly ready: boolean;
  readonly restarts: number;
  readonly desiredCount: number;
  readonly maxWindows: number;
  readonly error?: string;
}

export interface BrowserWorker {
  readonly index: number;
  readonly port: number;
  readonly maxWindows: number;
  readonly desiredCount: number;
  readonly hasCapacity: boolean;
  start(): Promise<void>;
  stop(): Promise<void>;
  open(config: WindowConfig): Promise<WindowSnapshot>;
  close(id: string): Promise<void>;
  closeAll(): Promise<void>;
  refresh(id: string): Promise<WindowSnapshot>;
  update(id: string, url: string): Promise<WindowSnapshot>;
  show(id: string, visible: boolean): Promise<WindowSnapshot>;
  status(): Promise<StatusReport>;
  supervisorStatus(error?: string): WorkerStatus;
}

export interface ElectronWorkerOptions {
  readonly index: number;
  readonly host: string;
  readonly port: number;
  readonly maxWindows: number;
  readonly requestTimeoutMs: number;
  readonly startupTimeoutMs: number;
  readonly restartDelayMs: number;
  readonly launcher: string;
  readonly userDataRoot: string;
  readonly parentEnv: NodeJS.ProcessEnv;
  readonly spawnProcess?: typeof spawn;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function errorForWorkerResponse(status: number, body: unknown): Error {
  const record = isRecord(body) ? body : {};
  const message =
    typeof record.error === 'string'
      ? record.error
      : `Electron worker returned HTTP ${String(status)}`;
  const code = typeof record.code === 'string' ? record.code : '';
  if (status === 400 || code === 'ValidationError') return new ValidationError(message);
  if (status === 404 || code === 'NotFoundError') return new NotFoundError(message);
  if (status === 409 || code === 'ConflictError') return new ConflictError(message);
  if (status === 503 || code === 'CapacityError') return new CapacityError(message);
  return new Error(message);
}

export class ElectronWorkerProcess implements BrowserWorker {
  public readonly index: number;
  public readonly port: number;
  public readonly maxWindows: number;
  private readonly opts: ElectronWorkerOptions;
  private readonly desired = new Map<string, DesiredWindow>();
  private readonly spawnProcess: typeof spawn;
  private child: ChildProcess | null = null;
  private ready = false;
  private stopping = false;
  private restarts = 0;
  private restartTimer: NodeJS.Timeout | null = null;
  private startPromise: Promise<void> | null = null;
  private operations: Promise<void> = Promise.resolve();

  constructor(opts: ElectronWorkerOptions) {
    this.opts = opts;
    this.index = opts.index;
    this.port = opts.port;
    this.maxWindows = opts.maxWindows;
    this.spawnProcess = opts.spawnProcess ?? spawn;
  }

  public get desiredCount(): number {
    return this.desired.size;
  }

  public get hasCapacity(): boolean {
    return this.desired.size < this.maxWindows;
  }

  public async start(): Promise<void> {
    if (this.ready && this.isAlive()) return;
    if (this.startPromise) return this.startPromise;
    this.stopping = false;
    const pending = this.startOnce();
    this.startPromise = pending;
    try {
      await pending;
    } finally {
      if (this.startPromise === pending) this.startPromise = null;
    }
  }

  public async stop(): Promise<void> {
    this.stopping = true;
    this.ready = false;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    const child = this.child;
    this.child = null;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()));
    child.kill('SIGTERM');
    const graceful = await Promise.race([
      exited.then(() => true),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 5000)),
    ]);
    if (!graceful && child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL');
      await exited;
    }
  }

  public async open(config: WindowConfig): Promise<WindowSnapshot> {
    if (!this.hasCapacity) {
      throw new CapacityError(`Electron worker ${String(this.index)} is at capacity`);
    }
    this.desired.set(config.id, { config, visible: false });
    return this.enqueue(async () => {
      try {
        await this.start();
        return await this.request<WindowSnapshot>('POST', '/window/open', config);
      } catch (err) {
        this.desired.delete(config.id);
        throw err;
      }
    });
  }

  public async close(id: string): Promise<void> {
    this.desired.delete(id);
    await this.enqueue(async () => {
      if (!this.ready || !this.isAlive()) return;
      try {
        await this.request('POST', '/window/close', { id });
      } catch (err) {
        if (!(err instanceof NotFoundError) && !(err instanceof WorkerUnavailableError)) throw err;
      }
    });
  }

  public async closeAll(): Promise<void> {
    this.desired.clear();
    await this.enqueue(async () => {
      if (!this.ready || !this.isAlive()) return;
      try {
        await this.request('GET', '/window/close/all');
      } catch (err) {
        if (!(err instanceof WorkerUnavailableError)) throw err;
      }
    });
  }

  public async refresh(id: string): Promise<WindowSnapshot> {
    return this.enqueue(async () => {
      await this.start();
      return this.request<WindowSnapshot>('POST', '/window/refresh', { id });
    });
  }

  public async update(id: string, url: string): Promise<WindowSnapshot> {
    const previous = this.desired.get(id);
    if (previous) {
      this.desired.set(id, { config: { ...previous.config, url }, visible: previous.visible });
    }
    return this.enqueue(async () => {
      await this.start();
      return this.request<WindowSnapshot>('POST', '/window/update', { id, url });
    });
  }

  public async show(id: string, visible: boolean): Promise<WindowSnapshot> {
    const previous = this.desired.get(id);
    if (previous) this.desired.set(id, { config: previous.config, visible });
    return this.enqueue(async () => {
      await this.start();
      return this.request<WindowSnapshot>('POST', '/window/show', { id, show: visible });
    });
  }

  public async status(): Promise<StatusReport> {
    if (!this.ready || !this.isAlive()) {
      throw new WorkerUnavailableError(`Electron worker ${String(this.index)} is not ready`);
    }
    return this.request<StatusReport>('GET', '/status');
  }

  public supervisorStatus(error?: string): WorkerStatus {
    return {
      index: this.index,
      port: this.port,
      alive: this.isAlive(),
      ready: this.ready,
      restarts: this.restarts,
      desiredCount: this.desired.size,
      maxWindows: this.maxWindows,
      ...(error === undefined ? {} : { error }),
    };
  }

  private async startOnce(): Promise<void> {
    if (!this.isAlive()) this.spawnChild();
    try {
      await this.waitUntilReady();
      this.ready = true;
    } catch (err) {
      this.ready = false;
      const child = this.child;
      if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      throw err;
    }
  }

  private spawnChild(): void {
    const workerName = `worker-${String(this.index).padStart(2, '0')}`;
    const env: NodeJS.ProcessEnv = {
      ...this.opts.parentEnv,
      DMA_BROWSER_WORKER_INDEX: String(this.index),
      DMA_BROWSER_REST_HOST: this.opts.host,
      DMA_BROWSER_REST_PORT: String(this.port),
      DMA_BROWSER_MAX_WINDOWS: String(this.maxWindows),
      DMA_BROWSER_AUTO_OPEN_URL: '',
      DMA_BROWSER_USER_DATA_DIR: path.join(this.opts.userDataRoot, workerName),
    };
    const child = this.spawnProcess(this.opts.launcher, [], {
      env,
      stdio: ['ignore', 'inherit', 'inherit'],
    });
    this.child = child;
    child.once('exit', () => {
      if (this.child !== child) return;
      this.child = null;
      this.ready = false;
      this.startPromise = null;
      if (!this.stopping) {
        this.restarts += 1;
        this.scheduleRestart();
      }
    });
    child.once('error', (err) => {
      console.error(`dma-browser worker ${String(this.index)} spawn error: ${err.message}`);
    });
  }

  private scheduleRestart(): void {
    if (this.restartTimer !== null || this.stopping) return;
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      void this.enqueue(async () => {
        try {
          await this.start();
          await this.restoreDesiredWindows();
        } catch (err) {
          console.error(
            `dma-browser worker ${String(this.index)} restart failed: ${
              err instanceof Error ? err.message : String(err)
            }`,
          );
          this.scheduleRestart();
        }
      });
    }, this.opts.restartDelayMs);
  }

  private async restoreDesiredWindows(): Promise<void> {
    for (const desired of this.desired.values()) {
      try {
        await this.request('POST', '/window/open', desired.config);
      } catch (err) {
        if (!(err instanceof ConflictError)) throw err;
      }
      if (desired.visible) {
        await this.request('POST', '/window/show', { id: desired.config.id, show: true });
      }
    }
  }

  private async waitUntilReady(): Promise<void> {
    const deadline = Date.now() + this.opts.startupTimeoutMs;
    let lastError = 'not ready';
    while (Date.now() < deadline) {
      if (!this.isAlive()) {
        throw new WorkerUnavailableError(
          `Electron worker ${String(this.index)} exited during startup`,
        );
      }
      try {
        await this.request<StatusReport>('GET', '/status');
        return;
      } catch (err) {
        lastError = err instanceof Error ? err.message : String(err);
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 100));
    }
    throw new WorkerUnavailableError(
      `Electron worker ${String(this.index)} did not become ready: ${lastError}`,
    );
  }

  private isAlive(): boolean {
    return this.child !== null && this.child.exitCode === null && this.child.signalCode === null;
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.operations.then(operation, operation);
    this.operations = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private request<T = unknown>(method: string, requestPath: string, body?: unknown): Promise<T> {
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    return new Promise<T>((resolve, reject) => {
      const req = http.request(
        {
          host: this.opts.host,
          port: this.opts.port,
          path: requestPath,
          method,
          headers:
            payload === null
              ? undefined
              : {
                  'content-type': 'application/json',
                  'content-length': String(payload.length),
                },
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on('data', (chunk: Buffer | string) =>
            chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk),
          );
          res.on('end', () => {
            const text = Buffer.concat(chunks).toString('utf8');
            let parsed: unknown = {};
            if (text.length > 0) {
              try {
                parsed = JSON.parse(text) as unknown;
              } catch {
                reject(new Error(`Electron worker returned invalid JSON for ${requestPath}`));
                return;
              }
            }
            const status = res.statusCode ?? 500;
            if (status < 200 || status >= 300) {
              reject(errorForWorkerResponse(status, parsed));
              return;
            }
            resolve(parsed as T);
          });
        },
      );
      req.setTimeout(this.opts.requestTimeoutMs, () => {
        req.destroy(new Error(`request timed out after ${String(this.opts.requestTimeoutMs)} ms`));
      });
      req.once('error', (err) => {
        reject(
          new WorkerUnavailableError(
            `Electron worker ${String(this.index)} request failed: ${err.message}`,
          ),
        );
      });
      if (payload !== null) req.write(payload);
      req.end();
    });
  }
}
