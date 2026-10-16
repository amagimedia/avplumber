import { app } from 'electron';
import { Application } from './Application';

function fail(message: string): never {
  console.error(`dma-browser fatal: ${message}`);
  process.exit(1);
}

if (process.platform !== 'linux') {
  fail(
    `dma-browser only supports Linux (fdpass + dmabuf are Linux-specific). Current platform: ${process.platform}`,
  );
}

const application = new Application();
// Chromium switches must be applied before app is ready.
application.applyHwAccelSwitches();

app.on('render-process-gone', (_event, _wc, details) => {
  console.error('render-process-gone:', details);
});

app.on('child-process-gone', (_event, details) => {
  console.error('child-process-gone:', details);
});

app
  .whenReady()
  .then(() => application.start())
  .catch((err: unknown) => {
    fail(`startup failed: ${err instanceof Error ? err.message : String(err)}`);
  });

app.on('window-all-closed', () => {
  // Do not quit; the REST server keeps the process alive and may open more windows.
});

const shutdown = (): void => {
  application
    .shutdown()
    .catch((err: unknown) => {
      console.error('shutdown error:', err);
    })
    .finally(() => {
      app.quit();
    });
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

process.on('uncaughtException', (err) => {
  console.error('uncaughtException:', err);
});

process.on('unhandledRejection', (err) => {
  console.error('unhandledRejection:', err);
});
