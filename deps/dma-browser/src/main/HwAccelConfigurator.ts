import { envFlag, envList, envString, type Env } from './support/env';

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
 * Configures Chromium command-line switches for hardware-accelerated video
 * decoding and offscreen capture on Linux + patched Electron/Chromium.
 *
 * Reads `DMA_BROWSER_*` env vars. The default feature set expects the
 * NVIDIA dmabuf patches from the S3 Electron 41 / Chrome 146 build.
 */
export class HwAccelConfigurator {
  // RenderableMappableSharedImageForceScanout is the offscreen-capture / NVIDIA
  // GBM SharedImage feature (NOT VAAPI). It forces renderable mappable
  // SharedImages to SCANOUT, which — together with the GBM linear shim — is what
  // lets Chromium allocate an exportable dmabuf on NVIDIA. Without it,
  // gbm_bo_create fails ("Cannot create bo ... usage=Scanout") and the GPU
  // process crashes. It must stay enabled even with no video decode (this
  // matches electron-hwaccel main.js, which enables it unconditionally). The
  // VAAPI/video-decode features are intentionally omitted here and explicitly
  // disabled below (a graphics overlay has no <video>).
  private static readonly ENABLED_FEATURES_BASE: readonly string[] = [
    'RenderableMappableSharedImageForceScanout',
  ];

  private static readonly DISABLED_FEATURES_BASE: readonly string[] = [
    'CompressionDictionaryTransport',
    'DefaultANGLEVulkan',
    'DrmOverlayManager',
    'SharedDictionaryCache',
    'VideoDecodeBatching',
    'Vulkan',
    'VulkanFromANGLE',
    // No hardware video decode for a graphics overlay: explicitly disable the
    // VAAPI/decode stack so the decode path is off (matches electron-hwaccel
    // main.js when VAAPI is disabled). These are inert without <video> but are
    // disabled so logs reflect the real (software) decode path.
    'AcceleratedVideoDecoder',
    'AcceleratedVideoDecodeLinuxGL',
    'VaapiOnNvidiaGPUs',
    'VaapiIgnoreDriverChecks',
    'UseChromeOSDirectVideoDecoder',
    'VaapiNvidiaSyncSurfaceBeforeOutput',
    'VaapiNvidiaResetRecreatesDecoder',
    'VaapiNvidiaSerializeDecodeLifecycle',
    'VaapiNvidiaDropStaleDecoderFrames',
    'VaapiNvidiaRtcResetDecoderAfterFrameGap',
    'VaapiNvidiaOutputBufferTuning',
    'ReduceHardwareVideoDecoderBuffers',
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
      this.appendSwitch(
        'use-angle',
        envString(this.env, 'DMA_BROWSER_ANGLE_BACKEND', 'gl-egl'),
      );
    }

    this.appendSwitch('disable-vulkan');
    this.appendSwitch('disable-hardware-overlays');
    this.appendSwitch('ignore-gpu-blocklist');
    // No hardware video decode needed for a graphics overlay; force software
    // decode so any stray <video> never hits the NVIDIA VAAPI path.
    this.appendSwitch('disable-accelerated-video-decode');

    const enabled = HwAccelConfigurator.dedupe([
      ...HwAccelConfigurator.ENABLED_FEATURES_BASE,
      ...envList(this.env, 'DMA_BROWSER_CHROMIUM_EXTRA_FEATURES'),
    ]);
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
          'gl_context*=3',
          'gl_display*=3',
          'gl_surface*=3',
          'gl_factory*=3',
          '*egl*=3',
          '*angle*=3',
          'gpu_init*=2',
          'gpu_info_collector*=2',
          'sandbox_linux*=2',
          '*video_decoder_pipeline*=2',
          'video_decoder_pipeline=1',
          '*vaapi_video_decoder*=3',
          'vaapi_video_decoder=3',
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
