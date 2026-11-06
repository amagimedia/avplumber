import { randomUUID } from 'node:crypto';
import { session } from 'electron';
import type { WebContents, WebPreferences } from 'electron';

/** Renderer restrictions for one controller-owned loopback scene, not an OS jail. */
export class IsolatedScenePolicy {
  public readonly url: string;
  private readonly origin: string;
  private readonly socketOrigin: string;

  constructor(value: string) {
    const parsed = new URL(value);
    if (
      parsed.protocol !== 'http:' ||
      parsed.hostname !== '127.0.0.1' ||
      !parsed.port ||
      parsed.username ||
      parsed.password ||
      parsed.pathname !== '/' ||
      parsed.search ||
      parsed.hash ||
      value !== parsed.href
    ) {
      throw new Error('Isolated scene must be a canonical loopback HTTP root');
    }
    this.url = parsed.href;
    this.origin = parsed.origin;
    this.socketOrigin = `ws://${parsed.host}`;
  }

  public allows(value: string, resourceType: string, method: string): boolean {
    try {
      if (method !== 'GET' || ['subFrame', 'object'].includes(resourceType)) return false;
      const parsed = new URL(value);
      if (parsed.username || parsed.password || parsed.hash) return false;
      if (resourceType === 'mainFrame') return value === this.url;
      if (resourceType === 'webSocket') {
        return (
          parsed.origin === this.socketOrigin && parsed.pathname === '/state' && !parsed.search
        );
      }
      if (parsed.protocol === 'data:') return ['image', 'font'].includes(resourceType);
      return (
        parsed.origin === this.origin &&
        /^\/(runtime|packages|scene-packages)\//.test(parsed.pathname)
      );
    } catch {
      return false;
    }
  }

  public preferences(): WebPreferences {
    const isolated = session.fromPartition(`scene-${randomUUID()}`, { cache: false });
    isolated.setPermissionRequestHandler((_wc, _permission, callback) => callback(false));
    isolated.setPermissionCheckHandler(() => false);
    isolated.on('will-download', (event) => event.preventDefault());
    isolated.webRequest.onBeforeRequest((details, callback) => {
      callback({ cancel: !this.allows(details.url, details.resourceType, details.method) });
    });
    return {
      session: isolated,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      nodeIntegrationInWorker: false,
      nodeIntegrationInSubFrames: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
      devTools: false,
      additionalArguments: [],
    };
  }

  public attach(contents: WebContents): void {
    contents.setWindowOpenHandler(() => ({ action: 'deny' }));
    contents.on('will-navigate', (event) => event.preventDefault());
    contents.on('will-frame-navigate', (event) => event.preventDefault());
    contents.on('will-redirect', (event) => event.preventDefault());
    contents.on('will-attach-webview', (event) => event.preventDefault());
    contents.setWebRTCIPHandlingPolicy('disable_non_proxied_udp');
  }
}
