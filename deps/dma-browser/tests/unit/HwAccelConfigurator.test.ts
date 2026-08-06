import { describe, expect, it } from 'vitest';
import {
  HwAccelConfigurator,
  type AppliedSwitch,
  type CommandLineLike,
} from '../../src/main/HwAccelConfigurator';

function recorder(): {
  cl: CommandLineLike;
  calls: AppliedSwitch[];
} {
  const calls: AppliedSwitch[] = [];
  const cl: CommandLineLike = {
    appendSwitch(name: string, value?: string) {
      calls.push(value === undefined ? { name } : { name, value });
    },
  };
  return { cl, calls };
}

function findSwitch(calls: AppliedSwitch[], name: string): AppliedSwitch | undefined {
  return calls.find((c) => c.name === name);
}

describe('HwAccelConfigurator.apply', () => {
  it('applies the baseline switches with defaults', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({}, cl).apply();

    const names = calls.map((c) => c.name);
    expect(names).toEqual(
      expect.arrayContaining([
        'enable-gpu',
        'no-sandbox',
        'run-all-compositor-stages-before-draw',
        'use-gl',
        'use-angle',
        'disable-vulkan',
        'disable-hardware-overlays',
        'ignore-gpu-blocklist',
        'enable-features',
        'disable-features',
        'high-dpi-support',
        'force-device-scale-factor',
        'disable-http-cache',
        'autoplay-policy',
        'disable-web-security',
      ]),
    );

    expect(findSwitch(calls, 'use-gl')?.value).toBe('angle');
    expect(findSwitch(calls, 'use-angle')?.value).toBe('gl-egl');
    expect(findSwitch(calls, 'autoplay-policy')?.value).toBe('no-user-gesture-required');
  });

  it('uses DMA_BROWSER_GL_BACKEND and skips use-angle for non-angle backends', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({ DMA_BROWSER_GL_BACKEND: 'egl' }, cl).apply();
    expect(findSwitch(calls, 'use-gl')?.value).toBe('egl');
    expect(findSwitch(calls, 'use-angle')).toBeUndefined();
  });

  it('honors DMA_BROWSER_ANGLE_BACKEND when angle is selected', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({ DMA_BROWSER_ANGLE_BACKEND: 'gl-swiftshader' }, cl).apply();
    expect(findSwitch(calls, 'use-angle')?.value).toBe('gl-swiftshader');
  });

  it('enables offscreen capture and disables hardware video decoding', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({}, cl).apply();
    const enabled = findSwitch(calls, 'enable-features')?.value ?? '';
    const disabled = findSwitch(calls, 'disable-features')?.value ?? '';

    expect(enabled.split(',')).toEqual(
      expect.arrayContaining([
        'RenderableMappableSharedImageForceScanout',
      ]),
    );
    expect(disabled.split(',')).toEqual(
      expect.arrayContaining([
        'AcceleratedVideoDecoder',
        'AcceleratedVideoDecodeLinuxGL',
        'VaapiOnNvidiaGPUs',
        'VaapiIgnoreDriverChecks',
        'UseChromeOSDirectVideoDecoder',
        'VaapiNvidiaSyncSurfaceBeforeOutput',
        'VaapiNvidiaOutputBufferTuning',
        'ReduceHardwareVideoDecoderBuffers',
      ]),
    );
    expect(findSwitch(calls, 'disable-accelerated-video-decode')).toBeDefined();
  });

  it('does not enable diagnostic or high-latency NVIDIA patch features by default', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({}, cl).apply();
    const enabled = findSwitch(calls, 'enable-features')?.value ?? '';
    for (const banned of [
      'VaapiNvidiaResetRecreatesDecoder',
      'VaapiNvidiaSyncSurfaceAfterEndPicture',
      'VaapiNvidiaSurfaceReuseChecks',
      'VaapiNvidiaDmabufFrameLogging',
    ]) {
      expect(enabled).not.toContain(banned);
    }
  });

  it('appends safe extra features without re-enabling disabled video decode', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator(
      {
        DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: 'AcceleratedVideoDecoder, MyCustomFeature',
        DMA_BROWSER_CHROMIUM_DISABLE_FEATURES: 'Vulkan, AnotherFeature',
      },
      cl,
    ).apply();
    const enabled = (findSwitch(calls, 'enable-features')?.value ?? '').split(',');
    const disabled = (findSwitch(calls, 'disable-features')?.value ?? '').split(',');
    expect(enabled).toContain('MyCustomFeature');
    expect(enabled).not.toContain('AcceleratedVideoDecoder');
    expect(disabled).toContain('AcceleratedVideoDecoder');
    expect(disabled).toContain('AnotherFeature');
    expect(disabled.filter((f) => f === 'Vulkan')).toHaveLength(1);
  });

  it('emits verbose logging switches when DMA_BROWSER_CHROMIUM_VLOG=1', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({ DMA_BROWSER_CHROMIUM_VLOG: '1' }, cl).apply();
    expect(findSwitch(calls, 'enable-logging')?.value).toBe('stderr');
    expect(findSwitch(calls, 'v')?.value).toBe('1');
    expect(findSwitch(calls, 'vmodule')?.value).toContain('gl_context');
    expect(findSwitch(calls, 'vmodule')?.value).not.toContain('vaapi_video_decoder');
  });

  it('does not emit vlog switches by default', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({}, cl).apply();
    expect(findSwitch(calls, 'enable-logging')).toBeUndefined();
    expect(findSwitch(calls, 'v')).toBeUndefined();
  });
});
