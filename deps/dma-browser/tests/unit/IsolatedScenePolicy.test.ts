import { describe, expect, it, vi } from 'vitest';
import { IsolatedScenePolicy } from '../../src/main/IsolatedScenePolicy';
import { HwAccelConfigurator } from '../../src/main/HwAccelConfigurator';
import { session } from 'electron';

vi.mock('electron', () => ({
  session: {
    fromPartition: vi.fn(() => ({
      setPermissionRequestHandler: vi.fn(),
      setPermissionCheckHandler: vi.fn(),
      on: vi.fn(),
      webRequest: { onBeforeRequest: vi.fn() },
    })),
  },
}));

const root = 'http://127.0.0.1:43123/';
describe('isolated scene policy', () => {
  it.each([
    'https://127.0.0.1:43123/',
    'http://localhost:43123/',
    'http://127.0.0.1/',
    root + '?other=1',
    root + '#x',
    root + 'other',
    'http://user@127.0.0.1:43123/',
    'http://2130706433:43123/',
    'file:///etc/passwd',
  ])('rejects scene identity %s', (value) => {
    expect(() => new IsolatedScenePolicy(value)).toThrow();
  });
  it.each([
    [root, 'mainFrame', true],
    [root + 'runtime/scene-client.mjs', 'script', true],
    [root + 'packages/0/main.mjs', 'script', true],
    [root + 'scene-packages/1/0/logo.svg', 'image', true],
    ['ws://127.0.0.1:43123/state', 'webSocket', true],
    ['data:image/png;base64,AA==', 'image', true],
    ['data:text/javascript,alert(1)', 'script', false],
    [root, 'subFrame', false],
    [root + 'packages/0/page.html', 'mainFrame', false],
    [root + 'state', 'xhr', false],
    ['http://127.0.0.1:9009/windows', 'xhr', false],
    ['http://169.254.169.254/', 'xhr', false],
    ['https://example.com/image.png', 'image', false],
    ['file:///etc/passwd', 'xhr', false],
    ['ws://127.0.0.1:43124/state', 'webSocket', false],
    ['ws://127.0.0.1:43123/other', 'webSocket', false],
    [root + 'packages/0/../../../state', 'xhr', false],
  ])('restricts %s (%s)', (url, kind, allowed) => {
    const policy = new IsolatedScenePolicy(root);
    expect(policy.allows(String(url), String(kind), 'GET')).toBe(allowed);
    expect(policy.allows(String(url), String(kind), 'POST')).toBe(false);
  });
  it('creates private sessions without preload, Node, permissions or insecure preferences', () => {
    const policy = new IsolatedScenePolicy(root);
    const first = policy.preferences();
    policy.preferences();
    const calls = vi.mocked(session.fromPartition).mock.calls;
    expect(calls.at(-1)?.[0]).not.toBe(calls.at(-2)?.[0]);
    expect(calls.at(-1)?.[0]).not.toContain('persist:');
    expect(first).toMatchObject({
      sandbox: true,
      contextIsolation: true,
      webSecurity: true,
      nodeIntegration: false,
      webviewTag: false,
      devTools: false,
      additionalArguments: [],
    });
    expect(first.preload).toBeUndefined();
    const active = vi.mocked(session.fromPartition).mock.results.at(-2)?.value;
    expect(active.setPermissionCheckHandler.mock.calls[0][0]()).toBe(false);
    const callback = vi.fn();
    active.setPermissionRequestHandler.mock.calls[0][0](null, 'camera', callback);
    expect(callback).toHaveBeenCalledWith(false);
    active.webRequest.onBeforeRequest.mock.calls[0][0](
      { url: 'file:///etc/passwd', resourceType: 'xhr', method: 'GET' },
      callback,
    );
    expect(callback).toHaveBeenLastCalledWith({ cancel: true });
  });
  it('keeps unsafe switches out of isolated launches', () => {
    const flags = new HwAccelConfigurator(
      { DMA_BROWSER_ISOLATED_SCENE_URL: root },
      { appendSwitch: vi.fn() },
    ).apply();
    expect(flags.map((flag) => flag.name)).not.toContain('no-sandbox');
    expect(flags.map((flag) => flag.name)).not.toContain('disable-web-security');
    expect(() =>
      new HwAccelConfigurator(
        {
          DMA_BROWSER_ISOLATED_SCENE_URL: root,
          DMA_BROWSER_CHROMIUM_DISABLE_FEATURES: 'Something',
        },
        { appendSwitch: vi.fn() },
      ).apply(),
    ).toThrow();
  });
});

it('denies navigation, redirects, webviews, popups and direct UDP WebRTC', () => {
  const contents = {
    setWindowOpenHandler: vi.fn(),
    on: vi.fn(),
    setWebRTCIPHandlingPolicy: vi.fn(),
  };
  new IsolatedScenePolicy(root).attach(contents as unknown as Electron.WebContents);
  expect(contents.setWindowOpenHandler.mock.calls[0]?.[0]()).toEqual({ action: 'deny' });
  expect(contents.on.mock.calls.map((call) => String(call[0]))).toEqual([
    'will-navigate',
    'will-frame-navigate',
    'will-redirect',
    'will-attach-webview',
  ]);
  for (const [, handler] of contents.on.mock.calls) {
    const event = { preventDefault: vi.fn() };
    handler(event);
    expect(event.preventDefault).toHaveBeenCalledOnce();
  }
  expect(contents.setWebRTCIPHandlingPolicy).toHaveBeenCalledWith('disable_non_proxied_udp');
});
