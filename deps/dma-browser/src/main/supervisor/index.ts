import * as path from 'node:path';
import { RestServer } from '../rest/RestServer';
import { ProcessManager } from './ProcessManager';
import { readSupervisorConfig } from './config';
import { ElectronWorkerProcess } from './WorkerProcess';

async function main(): Promise<void> {
  const projectRoot = path.resolve(__dirname, '..', '..', '..');
  const config = readSupervisorConfig(process.env, projectRoot);
  const workers = Array.from(
    { length: config.workerCount },
    (_, index) =>
      new ElectronWorkerProcess({
        index,
        host: config.workerHost,
        port: config.workerBasePort + index,
        maxWindows: config.windowsPerWorker,
        requestTimeoutMs: config.requestTimeoutMs,
        startupTimeoutMs: config.startupTimeoutMs,
        restartDelayMs: config.restartDelayMs,
        launcher: config.launcher,
        userDataRoot: config.userDataRoot,
        parentEnv: process.env,
      }),
  );
  const manager = new ProcessManager(workers, config.maxWindows);
  const rest = new RestServer(manager, {
    host: config.publicHost,
    port: config.publicPort,
  });

  await manager.start();
  await rest.listen();
  console.error(
    `dma-browser supervisor listening on http://${config.publicHost}:${String(
      config.publicPort,
    )}; workers=${String(config.workerCount)} windowsPerWorker=${String(config.windowsPerWorker)}`,
  );
  if (config.autoOpen) await manager.open(config.autoOpen);

  let shuttingDown = false;
  const shutdown = (): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    void (async () => {
      await rest.close();
      await manager.stop();
    })()
      .catch((err: unknown) => {
        console.error(`dma-browser supervisor shutdown failed: ${String(err)}`);
        process.exitCode = 1;
      })
      .finally(() => process.exit());
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

void main().catch((err: unknown) => {
  console.error(
    `dma-browser supervisor fatal: ${err instanceof Error ? err.message : String(err)}`,
  );
  process.exit(1);
});
