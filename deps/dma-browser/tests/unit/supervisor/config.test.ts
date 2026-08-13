import { describe, expect, it } from 'vitest';
import { readSupervisorConfig } from '../../../src/main/supervisor/config';

describe('readSupervisorConfig', () => {
  it('configures sixteen one-window workers and one public REST endpoint', () => {
    const config = readSupervisorConfig(
      {
        DMA_BROWSER_PROCESS_COUNT: '16',
        DMA_BROWSER_WINDOWS_PER_PROCESS: '1',
        DMA_BROWSER_REST_HOST: '0.0.0.0',
        DMA_BROWSER_REST_PORT: '9009',
        DMA_BROWSER_WORKER_BASE_PORT: '9010',
      },
      '/opt/dma-browser',
    );
    expect(config.workerCount).toBe(16);
    expect(config.windowsPerWorker).toBe(1);
    expect(config.maxWindows).toBe(8);
    expect(config.publicPort).toBe(9009);
    expect(config.workerBasePort).toBe(9010);
    expect(config.launcher).toBe('/opt/dma-browser/bin/run.sh');
  });

  it('automatically groups the requested windows eight per Electron process', () => {
    const config = readSupervisorConfig(
      {
        DMA_BROWSER_MAX_WINDOWS: '20',
      },
      '/opt/dma-browser',
    );
    expect(config.workerCount).toBe(3);
    expect(config.windowsPerWorker).toBe(8);
    expect(config.maxWindows).toBe(20);
  });

  it('rejects an explicit process count that cannot cover the global capacity', () => {
    expect(() =>
      readSupervisorConfig(
        {
          DMA_BROWSER_MAX_WINDOWS: '20',
          DMA_BROWSER_PROCESS_COUNT: '2',
          DMA_BROWSER_WINDOWS_PER_PROCESS: '8',
        },
        '/opt/dma-browser',
      ),
    ).toThrow('must cover');
  });

  it('validates auto-open through the public window schema', () => {
    const config = readSupervisorConfig(
      {
        DMA_BROWSER_AUTO_OPEN_URL: 'https://example.com/',
        DMA_BROWSER_AUTO_OPEN_ID: 'overlay',
        DMA_BROWSER_AUTO_OPEN_WIDTH: '1920',
        DMA_BROWSER_AUTO_OPEN_HEIGHT: '1080',
        DMA_BROWSER_AUTO_OPEN_FPS: '60',
      },
      '/app',
    );
    expect(config.autoOpen).toMatchObject({
      id: 'overlay',
      width: 1920,
      height: 1080,
      fps: 60,
    });
  });

  it('rejects externally reachable worker endpoints and overlapping ports', () => {
    expect(() => readSupervisorConfig({ DMA_BROWSER_WORKER_HOST: '0.0.0.0' }, '/app')).toThrow(
      'loopback',
    );
    expect(() =>
      readSupervisorConfig(
        {
          DMA_BROWSER_REST_HOST: '0.0.0.0',
          DMA_BROWSER_REST_PORT: '9010',
          DMA_BROWSER_WORKER_BASE_PORT: '9010',
        },
        '/app',
      ),
    ).toThrow('overlaps');
  });
});
