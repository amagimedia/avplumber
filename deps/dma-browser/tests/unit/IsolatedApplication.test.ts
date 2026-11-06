import { afterEach, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { app } from 'electron';
import { Application } from '../../src/main/Application';
import { RestServer } from '../../src/main/rest/RestServer';
import { WindowManager } from '../../src/main/WindowManager';

vi.mock('electron', () => ({
  app: {
    enableSandbox: vi.fn(),
    commandLine: { appendSwitch: vi.fn(), hasSwitch: vi.fn(() => false) },
    getGPUFeatureStatus: vi.fn(() => ({})),
  },
}));
vi.mock('../../src/main/ManagedWindow', () => ({ ManagedWindow: vi.fn() }));
vi.mock('../../src/main/WindowManager', () => ({
  WindowManager: vi.fn(() => ({
    open: vi.fn(),
    closeAll: vi.fn(),
  })),
}));
vi.mock('../../src/main/rest/RestServer', () => ({
  RestServer: vi.fn(() => ({ listen: vi.fn(), close: vi.fn() })),
}));
vi.mock('../../src/main/support/Logger', () => ({
  Logger: vi.fn(() => ({ global: vi.fn(), closeAll: vi.fn() })),
}));
vi.mock('../../src/main/capture/FrameCaptureChannel', () => ({
  DEFAULT_RETAINED_FRAME_POOL_SIZE: 4,
}));
const root = 'http://127.0.0.1:43211/';
const env = {
  DMA_BROWSER_ISOLATED_SCENE_URL: root,
  DMA_BROWSER_AUTO_OPEN_URL: root,
  DMA_BROWSER_MAX_WINDOWS: '1',
  DMA_BROWSER_PROCESS_COUNT: '1',
};
afterEach(() => vi.clearAllMocks());

it('starts the fixed scene with sandboxing and no REST control endpoint', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'scene-policy-'));
  const application = new Application({ ...env, DMA_BROWSER_SOCKET_DIR: directory });
  try {
    application.applyHwAccelSwitches();
    expect(app.enableSandbox).toHaveBeenCalledOnce();
    await application.start();
    expect(RestServer).not.toHaveBeenCalled();
    const manager = vi.mocked(WindowManager).mock.results[0]?.value;
    expect(manager.open).toHaveBeenCalledWith(expect.objectContaining({ url: root, audio: false }));
    await application.shutdown();
    expect(manager.closeAll).toHaveBeenCalledOnce();
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

it.each([
  { DMA_BROWSER_AUTO_OPEN_URL: 'http://127.0.0.1:43212/' },
  { DMA_BROWSER_AUTO_OPEN_URL: '' },
  { DMA_BROWSER_AUTO_OPEN_AUDIO: '1' },
  { DMA_BROWSER_MAX_WINDOWS: '2' },
  { DMA_BROWSER_PROCESS_COUNT: '2' },
])('rejects an incompatible isolated window %s', (override) => {
  expect(() => new Application({ ...env, ...override })).toThrow();
  expect(WindowManager).not.toHaveBeenCalled();
});

it('rejects inherited unsafe flags before enabling the runtime', () => {
  vi.mocked(app.commandLine.hasSwitch).mockReturnValueOnce(true);
  expect(() => new Application(env).applyHwAccelSwitches()).toThrow('Unsafe isolated browser');
  expect(app.enableSandbox).not.toHaveBeenCalled();
});
