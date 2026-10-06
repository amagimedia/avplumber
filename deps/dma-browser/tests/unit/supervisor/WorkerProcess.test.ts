import type { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import { describe, expect, it } from 'vitest';
import type { WindowConfig } from '../../../src/main/config/WindowConfig';
import { ElectronWorkerProcess } from '../../../src/main/supervisor/WorkerProcess';

class FakeChild extends EventEmitter {
  public exitCode: number | null = null;
  public signalCode: NodeJS.Signals | null = null;

  public kill(signal: NodeJS.Signals): boolean {
    this.signalCode = signal;
    setImmediate(() => this.emit('exit', null, signal));
    return true;
  }
}

/** The Electron worker's REST API, where every page load takes `loadMs`. */
function fakeWorkerApi(loadMs: number) {
  const pages = { inFlight: 0, peak: 0, opened: [] as string[] };
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (chunk: Buffer) => (body += chunk.toString()));
    req.on('end', () => {
      const reply = (value: unknown): void => {
        res.setHeader('content-type', 'application/json');
        res.end(JSON.stringify(value));
      };
      if (req.url !== '/window/open') {
        reply({ windows: [], count: 0, maxWindows: 8 });
        return;
      }
      const config = JSON.parse(body) as WindowConfig;
      pages.inFlight += 1;
      pages.peak = Math.max(pages.peak, pages.inFlight);
      setTimeout(() => {
        pages.inFlight -= 1;
        pages.opened.push(config.id);
        reply(config);
      }, loadMs);
    });
  });
  return { server, pages };
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

describe('ElectronWorkerProcess restart', () => {
  it('reopens its pages four at a time', async () => {
    const { server, pages } = fakeWorkerApi(20);
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const children: FakeChild[] = [];
    const worker = new ElectronWorkerProcess({
      index: 0,
      host: '127.0.0.1',
      port: (server.address() as AddressInfo).port,
      maxWindows: 8,
      requestTimeoutMs: 1000,
      startupTimeoutMs: 1000,
      restartDelayMs: 10,
      launcher: 'electron',
      userDataRoot: '/tmp/dma-browser-test',
      parentEnv: {},
      spawnProcess: (() => {
        const child = new FakeChild();
        children.push(child);
        return child;
      }) as unknown as typeof spawn,
    });
    try {
      const ids = Array.from({ length: 8 }, (_, index) => config(index).id);
      await Promise.all(ids.map(async (_, index) => worker.open(config(index))));
      expect(pages.peak).toBe(1); // operator opens stay ordered per worker
      pages.peak = 0;
      pages.opened.length = 0;

      await worker.restart();
      expect(children).toHaveLength(2);
      expect(pages.peak).toBe(4);
      expect(pages.opened.sort()).toEqual(ids);
    } finally {
      await worker.stop();
      server.closeAllConnections();
      server.close();
    }
  });
});
