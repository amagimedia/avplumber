import * as fs from 'node:fs';
import * as path from 'node:path';
import { app } from 'electron';
import { HwAccelConfigurator } from './HwAccelConfigurator';
import { Logger } from './support/Logger';
import { ManagedWindow, type IManagedWindowFactory } from './ManagedWindow';
import { WindowManager } from './WindowManager';
import { AllowedDims } from './capture/AllowedDims';
import { DEFAULT_RETAINED_FRAME_POOL_SIZE } from './capture/FrameCaptureChannel';
import { RestServer } from './rest/RestServer';
import { envFlag, envInt, envList, envString, type Env } from './support/env';
import type { WindowConfig } from './config/WindowConfig';

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
      )}`,
    );
    try {
      this.logger.global(`gpuFeatureStatus=${JSON.stringify(app.getGPUFeatureStatus())}`);
    } catch (err) {
      this.logger.global(`gpuFeatureStatus unavailable: ${String(err)}`);
    }

    const logger = this.logger;
    const factory: IManagedWindowFactory = {
      create: (windowCfg: WindowConfig) =>
        new ManagedWindow({
          config: windowCfg,
          socketDir: this.cfg.socketDir,
          preloadPath: this.preloadPath(),
          logger,
          allowedDims: this.cfg.allowedDims,
          loadWatchdogMs: this.cfg.loadWatchdogMs,
          retainedFramePoolSize: this.cfg.dmabufPoolSize,
        }),
    };

    this.windowManager = new WindowManager(factory, { maxWindows: this.cfg.maxWindows });
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
    const dmabufPoolSize = envInt(
      this.env,
      'DMA_BROWSER_DMABUF_POOL_SIZE',
      DEFAULT_RETAINED_FRAME_POOL_SIZE,
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
