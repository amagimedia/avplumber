import * as fs from 'node:fs';
import * as path from 'node:path';
import { app } from 'electron';
import { HwAccelConfigurator } from './HwAccelConfigurator';
import { Logger } from './support/Logger';
import { ManagedWindow, type IManagedWindowFactory } from './ManagedWindow';
import { workerFramePhase } from './capture/FramePhase';
import { WindowManager } from './WindowManager';
import { AllowedDims } from './capture/AllowedDims';
import { RestServer } from './rest/RestServer';
import { envFlag, envInt, envList, envString, type Env } from './support/env';
import type { CaptureMode, WindowConfig } from './config/WindowConfig';

const SOCKET_DIR = '/tmp/dma-page';
const DEFAULT_ALLOWED_DIMS = ['1920x1080', '1280x720'];
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;

export interface ApplicationConfig {
  readonly socketDir: string;
  readonly host: string;
  readonly port: number;
  readonly maxWindows: number;
  readonly loadWatchdogMs: number;
  readonly allowedDims: AllowedDims;
  readonly dmabufPoolSize: number;
  /** The capacity of a window's buffer queue in Chromium (`DMA_BROWSER_EXPORT_BUFFERS`). */
  readonly exportBuffers: number;
  /** `zero-copy` when `DMA_BROWSER_ZERO_COPY_EXPORT` is on: no retained pool of our own. */
  readonly captureMode: CaptureMode;
  readonly framePhase: number | null;
}

export class Application {
  private readonly env: Env;
  private readonly cfg: ApplicationConfig;
  private readonly autoOpenConfig: WindowConfig | null;
  private logger: Logger | null = null;
  private restServer: RestServer | null = null;
  private windowManager: WindowManager | null = null;

  constructor(env: Env = process.env) {
    this.env = env;
    this.cfg = this.readConfig();
    this.autoOpenConfig = this.readAutoOpenConfig();
  }

  /** Apply Chromium switches. Must run BEFORE `app.whenReady()`. */
  public applyHwAccelSwitches(): void {
    new HwAccelConfigurator(this.env, app.commandLine).apply();
  }

  /** Start the REST server and prepare the window manager. */
  public async start(): Promise<void> {
    fs.mkdirSync(this.cfg.socketDir, { recursive: true });
    this.logger = new Logger(this.cfg.socketDir);
    this.logger.global(
      `dma-browser starting; host=${this.cfg.host} port=${String(this.cfg.port)} maxWindows=${String(
        this.cfg.maxWindows,
      )} allowedDims=${this.cfg.allowedDims.listForLogging()} dmabufPoolSize=${String(
        this.cfg.dmabufPoolSize,
      )} exportBuffers=${String(this.cfg.exportBuffers)} captureMode=${this.cfg.captureMode} framePhase=${String(
        this.cfg.framePhase,
      )}`,
    );
    if (this.cfg.captureMode === 'zero-copy') {
      this.logger.global(
        `zero-copy export: Chromium's queue of ${String(this.cfg.exportBuffers)} buffers per window ` +
          'bounds the frames out; frames are lent until the consumer acknowledges them and no ' +
          'retained pool of our own applies (ringSize is echoed, not enforced)',
      );
    }
    try {
      this.logger.global(`gpuFeatureStatus=${JSON.stringify(app.getGPUFeatureStatus())}`);
    } catch (err) {
      this.logger.global(`gpuFeatureStatus unavailable: ${String(err)}`);
    }

    const logger = this.logger;
    const factory: IManagedWindowFactory = {
      create: (windowCfg: WindowConfig) => {
        const zeroCopy = this.cfg.captureMode === 'zero-copy';
        if (zeroCopy && windowCfg.ringSize !== undefined && windowCfg.ringSize >= this.cfg.exportBuffers) {
          logger.global(
            `window ${windowCfg.id}: ringSize=${String(windowCfg.ringSize)} is not below ` +
              `DMA_BROWSER_EXPORT_BUFFERS=${String(this.cfg.exportBuffers)}; a mixer holding that many ` +
              'leaves Chromium no free buffer to paint into (use browser_frame_queue <= exportBuffers - 1)',
          );
        }
        return new ManagedWindow({
          config: windowCfg,
          socketDir: this.cfg.socketDir,
          preloadPath: this.preloadPath(),
          logger,
          allowedDims: this.cfg.allowedDims,
          loadWatchdogMs: this.cfg.loadWatchdogMs,
          retainedFramePoolSize: zeroCopy ? null : (windowCfg.ringSize ?? this.cfg.dmabufPoolSize),
          captureMode: this.cfg.captureMode,
          exportBuffers: this.cfg.exportBuffers,
          framePhase: this.cfg.framePhase,
        });
      },
    };

    this.windowManager = new WindowManager(factory, {
      maxWindows: this.cfg.maxWindows,
      captureMode: this.cfg.captureMode,
      exportBuffers: this.cfg.exportBuffers,
    });
    this.restServer = new RestServer(this.windowManager, {
      host: this.cfg.host,
      port: this.cfg.port,
    });
    await this.restServer.listen();
    this.logger.global(`REST server listening on http://${this.cfg.host}:${String(this.cfg.port)}`);

    if (this.autoOpenConfig) {
      this.logger.global(
        `auto-opening window id=${this.autoOpenConfig.id} url=${this.autoOpenConfig.url} ` +
          `size=${String(this.autoOpenConfig.width)}x${String(this.autoOpenConfig.height)} ` +
          `fps=${String(this.autoOpenConfig.fps)} audio=${String(this.autoOpenConfig.audio)}`,
      );
      await this.windowManager.open(this.autoOpenConfig);
    }
  }

  public async shutdown(): Promise<void> {
    if (this.restServer) {
      await this.restServer.close();
      this.restServer = null;
    }
    if (this.windowManager) {
      await this.windowManager.closeAll();
      this.windowManager = null;
    }
    if (this.logger) {
      this.logger.global('dma-browser shutdown complete');
      this.logger.closeAll();
      this.logger = null;
    }
  }

  private preloadPath(): string {
    return path.resolve(__dirname, '..', 'preload', 'index.js');
  }

  private readConfig(): ApplicationConfig {
    const host = envString(this.env, 'DMA_BROWSER_REST_HOST', '127.0.0.1');
    const port = envInt(this.env, 'DMA_BROWSER_REST_PORT', 9009, 1, 65535);
    const maxWindows = envInt(this.env, 'DMA_BROWSER_MAX_WINDOWS', 8, 1, Number.MAX_SAFE_INTEGER);
    const loadWatchdogMs = envInt(this.env, 'DMA_BROWSER_LOAD_WATCHDOG_MS', 30_000, 1000, 600_000);
    // One number sizes the chain: Chromium's buffer queue (through the zero-copy feature's
    // parameter) and, on the stock copy path, the default retained pool here.
    const exportBuffers = HwAccelConfigurator.exportBuffers(this.env);
    const captureMode: CaptureMode = HwAccelConfigurator.zeroCopy(this.env) ? 'zero-copy' : 'copy';
    const dmabufPoolSize = envInt(
      this.env,
      'DMA_BROWSER_DMABUF_POOL_SIZE',
      exportBuffers,
      1,
      64,
    );
    const dimsList = envList(this.env, 'DMA_BROWSER_ALLOWED_DIMS');
    const allowedDims = AllowedDims.fromList(dimsList.length > 0 ? dimsList : DEFAULT_ALLOWED_DIMS);
    return {
      socketDir: envString(this.env, 'DMA_BROWSER_SOCKET_DIR', SOCKET_DIR),
      host,
      port,
      maxWindows,
      loadWatchdogMs,
      allowedDims,
      dmabufPoolSize,
      exportBuffers,
      captureMode,
      framePhase: workerFramePhase(this.env),
    };
  }

  private readAutoOpenConfig(): WindowConfig | null {
    const url = envString(this.env, 'DMA_BROWSER_AUTO_OPEN_URL', '').trim();
    if (url.length === 0) {
      return null;
    }

    const id = envString(this.env, 'DMA_BROWSER_AUTO_OPEN_ID', 'overlay').trim();
    if (!ID_RE.test(id)) {
      throw new Error(
        'DMA_BROWSER_AUTO_OPEN_ID must match /^[A-Za-z0-9_-]{1,64}$/ ' +
          '(alphanumeric, dash, underscore; 1-64 chars)',
      );
    }
    this.validateAutoOpenUrl(url);

    return {
      id,
      url,
      width: envInt(this.env, 'DMA_BROWSER_AUTO_OPEN_WIDTH', 1080, 16, 7680),
      height: envInt(this.env, 'DMA_BROWSER_AUTO_OPEN_HEIGHT', 1920, 16, 7680),
      fps: envInt(this.env, 'DMA_BROWSER_AUTO_OPEN_FPS', 30, 1, 240),
      audio: envFlag(this.env, 'DMA_BROWSER_AUTO_OPEN_AUDIO', false),
    };
  }

  private validateAutoOpenUrl(raw: string): void {
    let parsed: URL;
    try {
      parsed = new URL(raw);
    } catch (err) {
      throw new Error(`DMA_BROWSER_AUTO_OPEN_URL is not a valid URL: ${String(err)}`);
    }
    if (!['http:', 'https:', 'file:', 'data:'].includes(parsed.protocol)) {
      throw new Error(
        `DMA_BROWSER_AUTO_OPEN_URL must use http/https/file/data protocol ` +
          `(got "${parsed.protocol}")`,
      );
    }
  }
}
