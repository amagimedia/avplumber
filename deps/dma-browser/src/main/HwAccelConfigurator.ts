import { envFlag, envInt, envList, envString, type Env } from './support/env';

/**
 * Minimal interface of Electron's `app.commandLine` that we depend on.
 * Defined here so unit tests can inject a mock without dragging in Electron.
 */
export interface CommandLineLike {
  appendSwitch(name: string, value?: string): void;
}

export interface AppliedSwitch {
  readonly name: string;
  readonly value?: string;
}

/**
 * Configures Chromium command-line switches for offscreen DMA-BUF capture on
 * Linux + Electron/Chromium.
 *
 * Reads `DMA_BROWSER_*` env vars. The two features only the patched Electron
 * knows (stock ignores them) have switches of their own, off by default: the
 * native-handle capture no longer waits for the GPU before `paint` and hands
 * the consumer an acquire fence instead, which is safe only with a consumer
 * that waits on it.
 */
export class HwAccelConfigurator {
  /** `DMA_BROWSER_NATIVE_HANDLE_CAPTURE`: native-handle textures, no wait, acquire fence. */
  private static readonly NATIVE_HANDLE_FEATURE = 'RenderableMappableSharedImageForceScanout';
  /** `DMA_BROWSER_ZERO_COPY_EXPORT`: the zero-copy export of such textures; needs the first. */
  private static readonly ZERO_COPY_FEATURE = 'AvpZeroCopyOffscreenExport';
  /**
   * `DMA_BROWSER_EXPORT_BUFFERS`: the capacity of a window's buffer queue, passed to the zero-copy
   * feature as its `buffers` parameter. Chromium owns exactly this many buffers per window, lends
   * at most this many, and does not paint a frame while all are out. 2 to 16, default 6.
   */
  public static readonly EXPORT_BUFFERS_DEFAULT = 6;
  public static readonly EXPORT_BUFFERS_MIN = 2;
  public static readonly EXPORT_BUFFERS_MAX = 16;
  private static readonly DISABLED_FEATURES_BASE: readonly string[] = [
    'CompressionDictionaryTransport',
    'DefaultANGLEVulkan',
    'DrmOverlayManager',
    'SharedDictionaryCache',
    'Vulkan',
    'VulkanFromANGLE',
  ];

  private readonly env: Env;
  private readonly commandLine: CommandLineLike;
  private readonly appliedSwitches: AppliedSwitch[] = [];

  constructor(env: Env, commandLine: CommandLineLike) {
    this.env = env;
    this.commandLine = commandLine;
  }

  public apply(): readonly AppliedSwitch[] {
    this.appendSwitch('enable-gpu');
    this.appendSwitch('no-sandbox');
    this.appendSwitch('run-all-compositor-stages-before-draw');

    const glBackend = envString(this.env, 'DMA_BROWSER_GL_BACKEND', 'angle');
    this.appendSwitch('use-gl', glBackend);
    if (glBackend === 'angle') {
      this.appendSwitch('use-angle', envString(this.env, 'DMA_BROWSER_ANGLE_BACKEND', 'gl-egl'));
    }

    this.appendSwitch('disable-hardware-overlays');
    this.appendSwitch('disable-accelerated-video-decode');
    this.appendSwitch('ignore-gpu-blocklist');
    // With the blocklist ignored, Chromium 144+ runs WebGPU on Vulkan through GL interop and
    // creates a Vulkan context in the GPU process even with Vulkan compositing off (Chromium has
    // no disable-vulkan switch). Recreating windows then leaves a Vulkan cleanup task that trips
    // AddCleanupTaskForSkiaFlush's CHECK on the shared-texture copy (electron/electron#54553).
    // The pages need no WebGPU; on OpenGL ES the interop and its Vulkan context are off (152+).
    if (envFlag(this.env, 'DMA_BROWSER_WEBGPU_OPENGLES', true)) {
      this.appendSwitch('use-webgpu-adapter', 'opengles');
    }
    // After three GPU-process crashes, each within five minutes of the last, Chromium quits the
    // worker ("GPU process isn't usable"); keep relaunching the GPU process instead.
    if (envFlag(this.env, 'DMA_BROWSER_DISABLE_GPU_CRASH_LIMIT', true)) {
      this.appendSwitch('disable-gpu-process-crash-limit');
    }

    const extra = envList(this.env, 'DMA_BROWSER_CHROMIUM_EXTRA_FEATURES');
    const enabled = HwAccelConfigurator.dedupe([...this.gatedFeatures(extra), ...extra]);
    if (enabled.length > 0) {
      this.appendSwitch('enable-features', enabled.join(','));
    }

    const disabled = HwAccelConfigurator.dedupe([
      ...HwAccelConfigurator.DISABLED_FEATURES_BASE,
      ...envList(this.env, 'DMA_BROWSER_CHROMIUM_DISABLE_FEATURES'),
    ]);
    this.appendSwitch('disable-features', disabled.join(','));

    if (envFlag(this.env, 'DMA_BROWSER_CHROMIUM_VLOG', false)) {
      this.appendSwitch('enable-logging', 'stderr');
      this.appendSwitch('v', '1');
      this.appendSwitch(
        'vmodule',
        [
          '*renderable_mappable_shared_image_video_frame_pool*=2',
          '*native_pixmap_frame_resource*=2',
          'native_pixmap_frame_resource=1',
        ].join(','),
      );
    }

    this.appendSwitch('high-dpi-support', '1');
    this.appendSwitch('force-device-scale-factor', '1');
    this.appendSwitch('disable-http-cache');
    this.appendSwitch('autoplay-policy', 'no-user-gesture-required');
    this.appendSwitch('disable-web-security');

    return this.appliedSwitches;
  }

  public getApplied(): readonly AppliedSwitch[] {
    return this.appliedSwitches;
  }

  /** The patched build's features their switches ask for; the free-form list may not name them. */
  private gatedFeatures(extra: readonly string[]): string[] {
    const switches: ReadonlyArray<readonly [string, string]> = [
      [HwAccelConfigurator.NATIVE_HANDLE_FEATURE, 'DMA_BROWSER_NATIVE_HANDLE_CAPTURE'],
      [HwAccelConfigurator.ZERO_COPY_FEATURE, 'DMA_BROWSER_ZERO_COPY_EXPORT'],
    ];
    for (const [feature, variable] of switches) {
      if (extra.some((item) => item === feature || item.startsWith(`${feature}:`))) {
        throw new Error(
          `${feature} is requested with ${variable}=1, not through ` +
            'DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: it is safe only with a consumer that waits ' +
            'on the acquire fence',
        );
      }
    }
    const zeroCopy = HwAccelConfigurator.zeroCopy(this.env);
    // The zero-copy export is a native-handle capture: the second switch implies the first.
    const nativeHandle = envFlag(this.env, 'DMA_BROWSER_NATIVE_HANDLE_CAPTURE', zeroCopy);
    if (zeroCopy && !nativeHandle) {
      throw new Error(
        'DMA_BROWSER_ZERO_COPY_EXPORT=1 needs DMA_BROWSER_NATIVE_HANDLE_CAPTURE unset or 1: ' +
          'the zero-copy export is a native-handle capture',
      );
    }
    const buffers = HwAccelConfigurator.exportBuffers(this.env);
    return [
      ...(nativeHandle ? [HwAccelConfigurator.NATIVE_HANDLE_FEATURE] : []),
      ...(zeroCopy ? [`${HwAccelConfigurator.ZERO_COPY_FEATURE}:buffers/${String(buffers)}`] : []),
    ];
  }

  /** `DMA_BROWSER_EXPORT_BUFFERS`, 2-16, default 6. */
  public static exportBuffers(env: Env): number {
    return envInt(
      env,
      'DMA_BROWSER_EXPORT_BUFFERS',
      HwAccelConfigurator.EXPORT_BUFFERS_DEFAULT,
      HwAccelConfigurator.EXPORT_BUFFERS_MIN,
      HwAccelConfigurator.EXPORT_BUFFERS_MAX,
    );
  }

  /** `DMA_BROWSER_ZERO_COPY_EXPORT`: the zero-copy export is on (the patched build's queue). */
  public static zeroCopy(env: Env): boolean {
    return envFlag(env, 'DMA_BROWSER_ZERO_COPY_EXPORT', false);
  }

  private appendSwitch(name: string, value?: string): void {
    if (value === undefined) {
      this.commandLine.appendSwitch(name);
      this.appliedSwitches.push({ name });
    } else {
      this.commandLine.appendSwitch(name, value);
      this.appliedSwitches.push({ name, value });
    }
  }

  private static dedupe(items: readonly string[]): readonly string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const item of items) {
      if (!seen.has(item)) {
        seen.add(item);
        out.push(item);
      }
    }
    return out;
  }
}
