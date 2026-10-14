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
        'disable-hardware-overlays',
        'disable-accelerated-video-decode',
        'ignore-gpu-blocklist',
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
    expect(findSwitch(calls, 'use-webgpu-adapter')?.value).toBe('opengles');
    expect(findSwitch(calls, 'disable-gpu-process-crash-limit')).toBeDefined();
  });

  it('keeps WebGPU on Vulkan and the GPU crash limit when their flags are off', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator(
      { DMA_BROWSER_WEBGPU_OPENGLES: '0', DMA_BROWSER_DISABLE_GPU_CRASH_LIMIT: 'false' },
      cl,
    ).apply();
    expect(findSwitch(calls, 'use-webgpu-adapter')).toBeUndefined();
    expect(findSwitch(calls, 'disable-gpu-process-crash-limit')).toBeUndefined();
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

  it('does not enable vendor-specific features by default', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({}, cl).apply();
    expect(findSwitch(calls, 'enable-features')).toBeUndefined();
  });

  it('appends extra/disable features from env and dedupes', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator(
      {
        DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: 'MyCustomFeature, OtherFeature, MyCustomFeature',
        DMA_BROWSER_CHROMIUM_DISABLE_FEATURES: 'Vulkan, AnotherFeature',
      },
      cl,
    ).apply();
    const enabled = (findSwitch(calls, 'enable-features')?.value ?? '').split(',');
    const disabled = (findSwitch(calls, 'disable-features')?.value ?? '').split(',');
    expect(enabled).toContain('OtherFeature');
    expect(enabled.filter((f) => f === 'MyCustomFeature')).toHaveLength(1);
    expect(disabled).toContain('AnotherFeature');
    expect(disabled.filter((f) => f === 'Vulkan')).toHaveLength(1);
  });

  it('emits verbose logging switches when DMA_BROWSER_CHROMIUM_VLOG=1', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({ DMA_BROWSER_CHROMIUM_VLOG: '1' }, cl).apply();
    expect(findSwitch(calls, 'enable-logging')?.value).toBe('stderr');
    expect(findSwitch(calls, 'v')?.value).toBe('1');
    expect(findSwitch(calls, 'vmodule')?.value).toContain(
      'renderable_mappable_shared_image_video_frame_pool',
    );
  });

  it('does not emit vlog switches by default', () => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator({}, cl).apply();
    expect(findSwitch(calls, 'enable-logging')).toBeUndefined();
    expect(findSwitch(calls, 'v')).toBeUndefined();
  });

  const SCANOUT = 'RenderableMappableSharedImageForceScanout';
  const ZERO_COPY = 'AvpZeroCopyOffscreenExport:buffers/9';
  const enabledFeatures = (env: Record<string, string>): string[] => {
    const { cl, calls } = recorder();
    new HwAccelConfigurator(env, cl).apply();
    return (findSwitch(calls, 'enable-features')?.value ?? '').split(',').filter(Boolean);
  };

  it('requests the native-handle (no-wait) capture only with DMA_BROWSER_NATIVE_HANDLE_CAPTURE', () => {
    expect(enabledFeatures({})).toEqual([]);
    expect(enabledFeatures({ DMA_BROWSER_NATIVE_HANDLE_CAPTURE: '0' })).toEqual([]);
    expect(enabledFeatures({ DMA_BROWSER_NATIVE_HANDLE_CAPTURE: '1' })).toEqual([SCANOUT]);
  });

  it('DMA_BROWSER_ZERO_COPY_EXPORT adds the zero-copy feature on top of the native-handle one', () => {
    expect(enabledFeatures({ DMA_BROWSER_ZERO_COPY_EXPORT: '1' })).toEqual([SCANOUT, ZERO_COPY]);
    expect(
      enabledFeatures({ DMA_BROWSER_NATIVE_HANDLE_CAPTURE: '1', DMA_BROWSER_ZERO_COPY_EXPORT: '1' }),
    ).toEqual([SCANOUT, ZERO_COPY]);
    expect(enabledFeatures({ DMA_BROWSER_ZERO_COPY_EXPORT: '0' })).toEqual([]);
  });

  it('DMA_BROWSER_EXPORT_BUFFERS sizes the zero-copy feature parameter, clamped to 2-16', () => {
    expect(enabledFeatures({ DMA_BROWSER_ZERO_COPY_EXPORT: '1', DMA_BROWSER_EXPORT_BUFFERS: '5' })).toEqual([
      SCANOUT,
      'AvpZeroCopyOffscreenExport:buffers/5',
    ]);
    expect(enabledFeatures({ DMA_BROWSER_ZERO_COPY_EXPORT: '1', DMA_BROWSER_EXPORT_BUFFERS: '0' })).toEqual([
      SCANOUT,
      'AvpZeroCopyOffscreenExport:buffers/2',
    ]);
    expect(enabledFeatures({ DMA_BROWSER_ZERO_COPY_EXPORT: '1', DMA_BROWSER_EXPORT_BUFFERS: '99' })).toEqual([
      SCANOUT,
      'AvpZeroCopyOffscreenExport:buffers/16',
    ]);
    expect(HwAccelConfigurator.zeroCopy({ DMA_BROWSER_ZERO_COPY_EXPORT: '1' })).toBe(true);
    expect(HwAccelConfigurator.zeroCopy({})).toBe(false);
  });

  it('refuses zero-copy export with the native-handle capture switched off', () => {
    expect(() =>
      enabledFeatures({ DMA_BROWSER_NATIVE_HANDLE_CAPTURE: '0', DMA_BROWSER_ZERO_COPY_EXPORT: '1' }),
    ).toThrow(/DMA_BROWSER_NATIVE_HANDLE_CAPTURE/);
  });

  it.each([SCANOUT, 'AvpZeroCopyOffscreenExport', ZERO_COPY])('refuses %s smuggled in through DMA_BROWSER_CHROMIUM_EXTRA_FEATURES', (feature) => {
    expect(() => enabledFeatures({ DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: `Other,${feature}` })).toThrow(
      /DMA_BROWSER_NATIVE_HANDLE_CAPTURE|DMA_BROWSER_ZERO_COPY_EXPORT/,
    );
    expect(() =>
      enabledFeatures({ DMA_BROWSER_ZERO_COPY_EXPORT: '1', DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: feature }),
    ).toThrow();
  });
});
