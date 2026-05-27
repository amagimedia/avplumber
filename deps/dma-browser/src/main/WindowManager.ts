import type {
  IManagedWindow,
  IManagedWindowFactory,
} from './ManagedWindow';
import type { WindowConfig, WindowSnapshot } from './config/WindowConfig';
import { CapacityError, ConflictError, NotFoundError } from './rest/errors';

export interface WindowManagerOptions {
  readonly maxWindows: number;
}

export interface StatusReport {
  readonly windows: readonly WindowSnapshot[];
  readonly count: number;
  readonly maxWindows: number;
}

export class WindowManager {
  private readonly factory: IManagedWindowFactory;
  private readonly maxWindows: number;
  private readonly windows = new Map<string, IManagedWindow>();

  constructor(factory: IManagedWindowFactory, opts: WindowManagerOptions) {
    this.factory = factory;
    this.maxWindows = opts.maxWindows;
  }

  public async open(config: WindowConfig): Promise<WindowSnapshot> {
    if (this.windows.has(config.id)) {
      throw new ConflictError(`Window with id "${config.id}" is already open`);
    }
    if (this.windows.size >= this.maxWindows) {
      throw new CapacityError(
        `Window cap reached: ${String(this.maxWindows)}. Close one before opening another.`,
      );
    }
    const win = this.factory.create(config);
    this.windows.set(config.id, win);
    try {
      await win.create();
    } catch (err) {
      this.windows.delete(config.id);
      try {
        await win.destroy();
      } catch {
        // ignore
      }
      throw err;
    }
    return win.snapshot();
  }

  public async close(id: string): Promise<void> {
    const win = this.windows.get(id);
    if (!win) throw new NotFoundError(`No window with id "${id}"`);
    this.windows.delete(id);
    await win.destroy();
  }

  public async closeAll(): Promise<void> {
    const wins = Array.from(this.windows.values());
    this.windows.clear();
    await Promise.all(wins.map((w) => w.destroy()));
  }

  public refresh(id: string): WindowSnapshot {
    const win = this.requireWindow(id);
    win.refresh();
    return win.snapshot();
  }

  public update(id: string, url: string): WindowSnapshot {
    const win = this.requireWindow(id);
    win.update(url);
    return win.snapshot();
  }

  public show(id: string, visible: boolean): WindowSnapshot {
    const win = this.requireWindow(id);
    win.show(visible);
    return win.snapshot();
  }

  public status(): StatusReport {
    return {
      windows: Array.from(this.windows.values()).map((w) => w.snapshot()),
      count: this.windows.size,
      maxWindows: this.maxWindows,
    };
  }

  private requireWindow(id: string): IManagedWindow {
    const win = this.windows.get(id);
    if (!win) throw new NotFoundError(`No window with id "${id}"`);
    return win;
  }
}
