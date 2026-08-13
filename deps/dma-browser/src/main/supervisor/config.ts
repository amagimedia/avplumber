import * as path from 'node:path';
import { envFlag, envInt, envString, type Env } from '../support/env';
import type { WindowConfig } from '../config/WindowConfig';
import { ConfigService } from '../config/ConfigService';

export interface SupervisorConfig {
  readonly publicHost: string;
  readonly publicPort: number;
  readonly workerHost: string;
  readonly workerBasePort: number;
  readonly workerCount: number;
  readonly windowsPerWorker: number;
  readonly maxWindows: number;
  readonly requestTimeoutMs: number;
  readonly startupTimeoutMs: number;
  readonly restartDelayMs: number;
  readonly launcher: string;
  readonly userDataRoot: string;
  readonly autoOpen: WindowConfig | null;
}

export function readSupervisorConfig(env: Env, projectRoot: string): SupervisorConfig {
  const publicHost = envString(env, 'DMA_BROWSER_REST_HOST', '127.0.0.1');
  const publicPort = envInt(env, 'DMA_BROWSER_REST_PORT', 9009, 1, 65535);
  const workerHost = envString(env, 'DMA_BROWSER_WORKER_HOST', '127.0.0.1');
  const workerBasePort = envInt(env, 'DMA_BROWSER_WORKER_BASE_PORT', 9010, 1, 65535);
  const maxWindows = envInt(
    env,
    'DMA_BROWSER_MAX_WINDOWS',
    8,
    1,
    Number.MAX_SAFE_INTEGER,
  );
  const windowsPerWorker = envInt(env, 'DMA_BROWSER_WINDOWS_PER_PROCESS', 8, 1, 64);
  const automaticWorkerCount = Math.ceil(maxWindows / windowsPerWorker);
  const workerCount = envInt(
    env,
    'DMA_BROWSER_PROCESS_COUNT',
    automaticWorkerCount,
    1,
    64,
  );
  if (workerCount * windowsPerWorker < maxWindows) {
    throw new Error(
      'DMA_BROWSER_PROCESS_COUNT * DMA_BROWSER_WINDOWS_PER_PROCESS must cover DMA_BROWSER_MAX_WINDOWS',
    );
  }
  if (!['127.0.0.1', '::1', 'localhost'].includes(workerHost)) {
    throw new Error('DMA_BROWSER_WORKER_HOST must be a loopback address');
  }
  if (workerBasePort + workerCount - 1 > 65535) {
    throw new Error('DMA_BROWSER_WORKER_BASE_PORT range exceeds port 65535');
  }
  if (
    (workerHost === publicHost || publicHost === '0.0.0.0' || publicHost === '::') &&
    publicPort >= workerBasePort &&
    publicPort < workerBasePort + workerCount
  ) {
    throw new Error('Public REST port overlaps the private Electron worker port range');
  }
  return {
    publicHost,
    publicPort,
    workerHost,
    workerBasePort,
    workerCount,
    windowsPerWorker,
    maxWindows,
    requestTimeoutMs: envInt(env, 'DMA_BROWSER_WORKER_REQUEST_TIMEOUT_MS', 10_000, 100, 120_000),
    startupTimeoutMs: envInt(env, 'DMA_BROWSER_WORKER_STARTUP_TIMEOUT_MS', 90_000, 1000, 600_000),
    restartDelayMs: envInt(env, 'DMA_BROWSER_WORKER_RESTART_DELAY_MS', 1000, 100, 60_000),
    launcher: envString(
      env,
      'DMA_BROWSER_WORKER_LAUNCHER',
      path.join(projectRoot, 'bin', 'run.sh'),
    ),
    userDataRoot: envString(
      env,
      'DMA_BROWSER_WORKER_DATA_ROOT',
      path.join('/tmp', 'dma-browser-workers'),
    ),
    autoOpen: readAutoOpen(env),
  };
}

function readAutoOpen(env: Env): WindowConfig | null {
  const url = envString(env, 'DMA_BROWSER_AUTO_OPEN_URL', '').trim();
  if (url.length === 0) return null;
  return new ConfigService().validateWindowConfig({
    id: envString(env, 'DMA_BROWSER_AUTO_OPEN_ID', 'overlay').trim(),
    url,
    width: envInt(env, 'DMA_BROWSER_AUTO_OPEN_WIDTH', 1080, 16, 7680),
    height: envInt(env, 'DMA_BROWSER_AUTO_OPEN_HEIGHT', 1920, 16, 7680),
    fps: envInt(env, 'DMA_BROWSER_AUTO_OPEN_FPS', 30, 1, 240),
    audio: envFlag(env, 'DMA_BROWSER_AUTO_OPEN_AUDIO', false),
  });
}
