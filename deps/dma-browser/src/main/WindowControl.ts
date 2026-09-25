import type { WindowConfig, WindowSnapshot } from './config/WindowConfig';

export interface StatusReport {
  readonly windows: readonly WindowSnapshot[];
  readonly count: number;
  readonly maxWindows: number;
}

export interface WindowControl {
  open(config: WindowConfig): Promise<WindowSnapshot>;
  close(id: string): Promise<void>;
  closeAll(): Promise<void>;
  refresh(id: string): WindowSnapshot | Promise<WindowSnapshot>;
  update(id: string, url: string): WindowSnapshot | Promise<WindowSnapshot>;
  show(id: string, visible: boolean): WindowSnapshot | Promise<WindowSnapshot>;
  status(): StatusReport | Promise<StatusReport>;
  // The caller must reap its consumer first and supply all of its window ids.
  // Implementations must refuse to restart workers shared with other consumers.
  recover?(ids: readonly string[]): Promise<void>;
}
