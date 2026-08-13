import * as path from 'node:path';
import { BrowserWindow } from 'electron';
import { AudioCaptureChannel } from './capture/AudioCaptureChannel';
import { FrameCaptureChannel } from './capture/FrameCaptureChannel';
import type { AllowedDims } from './capture/AllowedDims';
import { LoadWatchdog } from './LoadWatchdog';
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

    this.watchdog = new LoadWatchdog({
      timeoutMs: this.opts.loadWatchdogMs,
      onTimeout: () => {
        log.write('load watchdog timeout; hard reloading');
        void this.hardReload();
      },
    });

    wc.on('did-finish-load', () => {
      this.watchdog?.clear();
    });

    wc.on('did-fail-load', (_e, errorCode, errorDesc, url, isMainFrame) => {
      log.write(
        `did-fail-load: code=${errorCode} desc=${errorDesc} url=${url} mainFrame=${String(isMainFrame)}`,
      );
      this.watchdog?.clear();
      void this.hardReload();
    });

    wc.on('console-message', (_e, level, message, line, source) => {
      log.write(`[renderer level=${level}] ${message} (${source}:${line})`);
    });

    const applyOverlays = (): void => {
      try {
        void wc.insertCSS(FULLSCREEN_CSS);
      } catch {
        // ignore
      }
      try {
        void wc.executeJavaScript(FULLSCREEN_JS, true);
      } catch {
        // ignore
      }
      try {
        void wc.executeJavaScript(AUTOPLAY_JS, true);
      } catch {
        // ignore
      }
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
    this.watchdog?.start();
    void this.loadUrl(url);
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
          lastPaintTsMs: channelStats.lastPaintTsMs,
        }
      : {
          paintCount: 0,
          droppedFrames: 0,
          droppedReasons: {},
          txFrameCount: 0,
          releasedFrameCount: 0,
          retainedFrameCount: 0,
          lastPaintTsMs: null,
        };
    return {
      id: this.id,
      url: this.config.url,
      width: this.config.width,
      height: this.config.height,
      fps: this.config.fps,
      audio: this.config.audio,
      visible: this.visible,
      stats,
    };
  }

  private async loadUrl(url: string): Promise<void> {
    if (!this.win) return;
    const sep = url.includes('?') ? '&' : '?';
    const urlToLoad = `${url}${sep}_cb=${Date.now().toString()}`;
    const extraHeaders = 'pragma: no-cache\ncache-control: no-cache, no-store, must-revalidate';
    try {
      await this.win.loadURL(urlToLoad, { extraHeaders });
    } catch (err) {
      this.opts.logger.forWindow(this.id).write(`loadURL failed: ${String(err)}`);
    }
  }

  private async hardReload(): Promise<void> {
    if (this.destroyed) return;
    const cfg = this.config;
    this.opts.logger.forWindow(this.id).write('hard reload begin');
    await this.destroy();
    // Recreate with the same config.
    this.destroyed = false;
    this.config = cfg;
    await this.create();
  }
}
