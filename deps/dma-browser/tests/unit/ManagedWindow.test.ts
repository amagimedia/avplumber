import { expect, it, vi } from 'vitest';
import { ManagedWindow } from '../../src/main/ManagedWindow';

vi.mock('electron', () => ({ BrowserWindow: vi.fn() }));
vi.mock('../../src/main/capture/FrameCaptureChannel', () => ({ FrameCaptureChannel: vi.fn() }));
vi.mock('../../src/main/capture/AudioCaptureChannel', () => ({ AudioCaptureChannel: vi.fn() }));

it('invalidates an active page without restarting its capture pool', () => {
  let painting = false;
  const webContents = {
    isPainting: () => painting,
    startPainting: vi.fn(() => { painting = true; }),
    stopPainting: vi.fn(() => { painting = false; }),
    invalidate: vi.fn(),
  };
  const window = Object.assign(Object.create(ManagedWindow.prototype), {
    win: { isDestroyed: () => false, webContents },
  }) as ManagedWindow;
  window.paint(true);
  window.paint(true);
  window.paint(true);
  expect(webContents.startPainting).toHaveBeenCalledTimes(1);
  expect(webContents.invalidate).toHaveBeenCalledTimes(3);
  window.paint(false);
  window.paint(false);
  expect(webContents.stopPainting).toHaveBeenCalledTimes(1);
  window.paint(true);
  expect(webContents.startPainting).toHaveBeenCalledTimes(2);
});
