import * as path from 'node:path';
import { BrowserWindow } from 'electron';
import { AudioCaptureChannel } from './capture/AudioCaptureChannel';
import { FrameCaptureChannel } from './capture/FrameCaptureChannel';
import { alignFramePhase, MONOTONIC_CLOCK } from './capture/FramePhase';
import type { AllowedDims } from './capture/AllowedDims';
import { LoadWatchdog } from './LoadWatchdog';
import { isPageLoadFailure, PageReloader } from './PageReloader';
import type { Logger } from './support/Logger';
import type { WindowConfig, WindowSnapshot, WindowStats } from './config/WindowConfig';

const FULLSCREEN_CSS = `
html, body, #root, #app {
  margin: 0 !important;
  padding: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  overflow: hidden !important;
}
#root, #app {
  background: rgba(0,0,0,0) !important;
}
* { overscroll-behavior: none !important; }
::-webkit-scrollbar { width: 0 !important; height: 0 !important; display: none !important; }
`.trim();

const RELOAD_MIN_DELAY_MS = 1000;
const RELOAD_MAX_DELAY_MS = 30_000;

const FULLSCREEN_JS = [
  'try {',
  '  document.documentElement.style.margin="0";',
  '  document.documentElement.style.padding="0";',
  '  document.documentElement.style.overflow="hidden";',
  '  document.documentElement.style.width="100vw";',
  '  document.documentElement.style.height="100vh";',
  '  document.documentElement.style.background="rgba(0,0,0,0)";',
  '  if (document.body) {',
  '    document.body.style.margin="0";',
  '    document.body.style.padding="0";',
  '    document.body.style.overflow="hidden";',
  '    document.body.style.width="100vw";',
  '    document.body.style.height="100vh";',
  '    document.body.style.background="rgba(0,0,0,0)";',
  '  }',
  '} catch (e) {}',
].join('\n');

const AUTOPLAY_JS = [
  'try {',
  '  const enable = (el) => {',
  '    if (!el) return;',
  '    try { el.crossOrigin = el.crossOrigin || "anonymous"; } catch (e) {}',
  '    el.autoplay = true;',
  '    el.muted = false;',
  '    const p = el.play && el.play();',
  '    if (p && typeof p.catch === "function") p.catch(() => {});',
  '  };',
  '  document.querySelectorAll("audio,video").forEach(enable);',
  '  const obs = new MutationObserver((muts) => {',
  '    muts.forEach((m) => {',
  '      m.addedNodes && m.addedNodes.forEach((n) => {',
  '        if (n && (n.tagName === "AUDIO" || n.tagName === "VIDEO")) enable(n);',
  '        if (n && n.querySelectorAll) n.querySelectorAll("audio,video").forEach(enable);',
  '      });',
  '    });',
  '  });',
  '  obs.observe(document.documentElement || document.body, { childList: true, subtree: true });',
  '} catch (e) {}',
].join('\n');

export interface ManagedWindowOptions {
  readonly config: WindowConfig;
  readonly socketDir: string;
  readonly preloadPath: string;
  readonly logger: Logger;
  readonly allowedDims: AllowedDims;
  readonly loadWatchdogMs: number;
  readonly retainedFramePoolSize: number;
  /** Where in the frame period this window's frames start (0 to 1); null keeps Chromium's timing. */
  readonly framePhase: number | null;
}

export interface IManagedWindow {
  readonly id: string;
  create(): Promise<void>;
  refresh(): void;
  update(url: string): void;
  show(visible: boolean): void;
  destroy(): Promise<void>;
  snapshot(): WindowSnapshot;
}

export interface IManagedWindowFactory {
  create(config: WindowConfig): IManagedWindow;
}

export class ManagedWindow implements IManagedWindow {
  public readonly id: string;
  private readonly opts: ManagedWindowOptions;
  private config: WindowConfig;
  private win: BrowserWindow | null = null;
  private frameChannel: FrameCaptureChannel | null = null;
  private audioChannel: AudioCaptureChannel | null = null;
  private watchdog: LoadWatchdog | null = null;
  private reloader: PageReloader | null = null;
  private visible = false;
  private destroyed = false;

  constructor(opts: ManagedWindowOptions) {
    this.opts = opts;
    this.config = opts.config;
    this.id = opts.config.id;
  }

  public async create(): Promise<void> {
    const log = this.opts.logger.forWindow(this.id);
    const sockPath = path.join(this.opts.socketDir, `${this.id}.sock`);
    const audioSockPath = path.join(this.opts.socketDir, `${this.id}-audio.sock`);

    this.frameChannel = new FrameCaptureChannel({
      socketPath: sockPath,
      allowedDims: this.opts.allowedDims,
      log,
      retainedFramePoolSize: this.opts.retainedFramePoolSize,
    });
    await this.frameChannel.start();

    if (this.config.audio) {
      this.audioChannel = new AudioCaptureChannel({
        windowId: this.id,
        socketPath: audioSockPath,
        log,
      });
      await this.audioChannel.start();
    }

    this.win = new BrowserWindow({
      width: this.config.width,
      height: this.config.height,
      transparent: true,
      backgroundColor: '#00000000',
      frame: false,
      show: false,
      webPreferences: {
        offscreen: { useSharedTexture: true },
        webSecurity: false,
        backgroundThrottling: false,
        sandbox: false,
        autoplayPolicy: 'no-user-gesture-required',
        preload: this.opts.preloadPath,
        additionalArguments: [
          `--window-id=${this.id}`,
          `--audio-enabled=${this.config.audio ? '1' : '0'}`,
        ],
      },
    });

    const wc = this.win.webContents;
    this.frameChannel.attach(wc);
    wc.setFrameRate(this.config.fps);

    const framePhase = this.opts.framePhase;
    if (framePhase !== null) {
      const target = {
        setFrameRate: (fps: number): void => {
          if (!wc.isDestroyed()) wc.setFrameRate(fps);
        },
      };
      // Every navigation: it can give the page a new compositor, back on Chromium's own timing.
      wc.on('did-navigate', () => {
        alignFramePhase(target, this.config.fps, framePhase, MONOTONIC_CLOCK)
          .then((aligned) => {
            if (!aligned) log.write('frame phase not set: timers woke too late');
          })
          .catch((err: unknown) => {
            log.write(`frame phase failed: ${String(err)}`);
          });
      });
    }

    // Recovery reloads the page only: the capture channel and its socket stay up, so the
    // consumer keeps its connection and the frames it holds.
    this.reloader = new PageReloader({
      minDelayMs: RELOAD_MIN_DELAY_MS,
      maxDelayMs: RELOAD_MAX_DELAY_MS,
      reload: () => {
        if (this.destroyed || !this.win || this.win.isDestroyed()) return;
        this.startLoading(this.config.url);
      },
    });
    const recover = (reason: string, hold = true): void => {
      if (this.destroyed) return;
      // Until a clean load, the error page and a half-loaded reload are not sent.
      if (hold && (this.config.holdLastFrame ?? true)) this.frameChannel?.hold(true);
      const delayMs = this.reloader?.failed() ?? null;
      const action =
        delayMs === null ? 'reload already pending' : `reloading page in ${delayMs} ms`;
      log.write(`${reason}; ${action}`);
      console.error(`dma-browser window ${this.id}: ${reason}; ${action}`);
    };

    this.watchdog = new LoadWatchdog({
      timeoutMs: this.opts.loadWatchdogMs,
      // A slow page is still painting its own content: reload it, but keep sending.
      onTimeout: () => recover('load watchdog timeout', false),
    });

    wc.on('did-finish-load', () => {
      this.watchdog?.clear();
      if (this.reloader?.loaded()) this.frameChannel?.hold(false);
    });

    wc.on('did-fail-load', (_e, errorCode, errorDesc, url, isMainFrame) => {
      const line = `did-fail-load: code=${errorCode} desc=${errorDesc} url=${url}`;
      if (isPageLoadFailure(errorCode, isMainFrame)) {
        this.watchdog?.clear();
        recover(line);
      } else {
        log.write(`${line} mainFrame=${String(isMainFrame)}`);
      }
    });

    wc.on('render-process-gone', (_e, details) => {
      recover(`render-process-gone: reason=${details.reason} exitCode=${details.exitCode}`);
    });

    wc.on('console-message', (_e, level, message, line, source) => {
      log.write(`[renderer level=${level}] ${message} (${source}:${line})`);
    });

    // Best effort: these reject asynchronously when the frame navigates away or its renderer dies.
    const applyOverlays = (): void => {
      const ignore = (): void => undefined;
      wc.insertCSS(FULLSCREEN_CSS).catch(ignore);
      wc.executeJavaScript(FULLSCREEN_JS, true).catch(ignore);
      wc.executeJavaScript(AUTOPLAY_JS, true).catch(ignore);
    };
    wc.on('dom-ready', applyOverlays);
    wc.on('did-navigate', applyOverlays);

    this.win.setBounds({ x: 0, y: 0, width: this.config.width, height: this.config.height });

    this.watchdog.start();
    await this.loadUrl(this.config.url);
  }

  public refresh(): void {
    if (!this.win || this.win.isDestroyed()) return;
    this.opts.logger.forWindow(this.id).write('refresh');
    this.reloader?.reset();
    try {
      this.win.webContents.reloadIgnoringCache();
    } catch {
      // ignore
    }
  }

  public update(url: string): void {
    if (!this.win || this.win.isDestroyed()) return;
    this.config = { ...this.config, url };
    this.opts.logger.forWindow(this.id).write(`update url=${url}`);
    this.reloader?.reset();
    this.startLoading(url);
  }

  public show(visible: boolean): void {
    if (!this.win || this.win.isDestroyed()) return;
    if (visible) {
      this.win.showInactive();
    } else {
      this.win.hide();
    }
    this.visible = visible;
  }

  public async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    this.opts.logger.forWindow(this.id).write('destroy');
    this.watchdog?.clear();
    this.watchdog = null;
    this.reloader?.reset();
    this.reloader = null;
    if (this.frameChannel) {
      await this.frameChannel.stop();
      this.frameChannel = null;
    }
    if (this.audioChannel) {
      await this.audioChannel.stop();
      this.audioChannel = null;
    }
    if (this.win && !this.win.isDestroyed()) {
      try {
        this.win.destroy();
      } catch {
        // ignore
      }
    }
    this.win = null;
    this.opts.logger.closeWindow(this.id);
  }

  public snapshot(): WindowSnapshot {
    const channelStats = this.frameChannel?.getStats();
    const stats: WindowStats = channelStats
      ? {
          paintCount: channelStats.paintCount,
          droppedFrames: channelStats.droppedFrames,
          droppedReasons: channelStats.droppedReasons,
          txFrameCount: channelStats.txFrameCount,
          releasedFrameCount: channelStats.releasedFrameCount,
          retainedFrameCount: channelStats.retainedFrameCount,
          quarantinedFrameCount: channelStats.quarantinedFrameCount,
          lastPaintTsMs: channelStats.lastPaintTsMs,
        }
      : {
          paintCount: 0,
          droppedFrames: 0,
          droppedReasons: {},
          txFrameCount: 0,
          releasedFrameCount: 0,
          retainedFrameCount: 0,
          quarantinedFrameCount: 0,
          lastPaintTsMs: null,
        };
    return {
      id: this.id,
      url: this.config.url,
      width: this.config.width,
      height: this.config.height,
      fps: this.config.fps,
      audio: this.config.audio,
      ringSize: this.opts.retainedFramePoolSize,
      holdLastFrame: this.config.holdLastFrame ?? true,
      visible: this.visible,
      stats,
    };
  }

  /** Loads without waiting: the watchdog and the did-*-load handlers track the outcome. */
  private startLoading(url: string): void {
    this.watchdog?.start();
    this.loadUrl(url).catch((err: unknown) => {
      this.opts.logger.forWindow(this.id).write(`loadURL failed: ${String(err)}`);
    });
  }

  private async loadUrl(url: string): Promise<void> {
    if (!this.win) return;
    const target = new URL(url);
    if (target.protocol === 'http:' || target.protocol === 'https:') {
      target.searchParams.set('_cb', Date.now().toString());
    }
    const urlToLoad = target.href;
    const extraHeaders = 'pragma: no-cache\ncache-control: no-cache, no-store, must-revalidate';
    try {
      await this.win.loadURL(urlToLoad, { extraHeaders });
    } catch (err) {
      this.opts.logger.forWindow(this.id).write(`loadURL failed: ${String(err)}`);
    }
  }
}
