import type { IpcBridge } from './IpcBridge';

const WORKLET_NAME = 'dma-collector';

/**
 * Inline AudioWorkletProcessor source (runs in the AudioWorklet global scope,
 * which is a separate JS world). Batches interleaved float32 PCM into chunks
 * of `batchSamples` frames and `postMessage`s them back to the renderer.
 */
const WORKLET_SOURCE = `
class DmaCollectorProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.batchSamples = (options && options.processorOptions && options.processorOptions.batchSamples) || 1024;
    this.buffer = new Float32Array(this.batchSamples * 2);
    this.offset = 0;
    this.channels = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input.length;
    if (ch !== this.channels) {
      this.channels = ch;
      this.buffer = new Float32Array(this.batchSamples * ch);
      this.offset = 0;
    }
    const frames = input[0].length;
    for (let i = 0; i < frames; i++) {
      for (let c = 0; c < ch; c++) {
        const lane = input[c];
        this.buffer[this.offset++] = lane ? lane[i] : 0;
      }
      if (this.offset >= this.buffer.length) {
        const chunk = this.buffer.slice(0);
        this.port.postMessage({ pcm: chunk.buffer, channels: ch, sampleRate, frames: this.batchSamples }, [chunk.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor(${JSON.stringify(WORKLET_NAME)}, DmaCollectorProcessor);
`;

export interface AudioCapturePipelineOptions {
  readonly windowId: string;
  readonly bridge: IpcBridge;
}

interface MediaSource {
  readonly source: MediaElementAudioSourceNode;
  readonly gain: GainNode;
}

export class AudioCapturePipeline {
  private readonly opts: AudioCapturePipelineOptions;
  private audioCtx: AudioContext | null = null;
  private collector: AudioWorkletNode | null = null;
  private masterGain: GainNode | null = null;
  private readonly sources = new Map<HTMLMediaElement, MediaSource>();
  private installed = false;

  constructor(opts: AudioCapturePipelineOptions) {
    this.opts = opts;
  }

  public install(): void {
    if (this.installed) return;
    this.installed = true;
    const ready = (): void => this.bindExisting();
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', ready, { once: true });
    } else {
      ready();
    }
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        m.addedNodes.forEach((n) => this.scanNode(n));
      }
    });
    const target = document.documentElement;
    if (target) observer.observe(target, { childList: true, subtree: true });
  }

  private bindExisting(): void {
    document.querySelectorAll<HTMLMediaElement>('audio,video').forEach((el) => this.bindElement(el));
  }

  private scanNode(node: Node): void {
    if (node instanceof HTMLAudioElement || node instanceof HTMLVideoElement) {
      this.bindElement(node);
    }
    if (node instanceof Element) {
      node.querySelectorAll<HTMLMediaElement>('audio,video').forEach((el) => this.bindElement(el));
    }
  }

  private bindElement(el: HTMLMediaElement): void {
    if (this.sources.has(el)) return;
    void this.ensureCollector()
      .then((ctx) => {
        try {
          const source = ctx.createMediaElementSource(el);
          const gain = ctx.createGain();
          gain.gain.value = 1.0;
          source.connect(gain);
          if (this.masterGain) gain.connect(this.masterGain);
          // Maintain the audio passing through to the speakers in case the
          // page expects to be heard locally. Harmless when no output device.
          gain.connect(ctx.destination);
          this.sources.set(el, { source, gain });
        } catch {
          // Some media elements (e.g. cross-origin without permission) cannot
          // be tapped. Ignore — the page still plays normally.
        }
      })
      .catch(() => {
        // ignore
      });
  }

  private async ensureCollector(): Promise<AudioContext> {
    if (this.audioCtx && this.collector && this.masterGain) return this.audioCtx;
    const ctx = new AudioContext();
    this.audioCtx = ctx;
    const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const collector = new AudioWorkletNode(ctx, WORKLET_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { batchSamples: 1024 },
    });
    collector.port.onmessage = (event: MessageEvent) => this.handleWorkletMessage(event);
    const masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;
    masterGain.connect(collector);
    this.collector = collector;
    this.masterGain = masterGain;
    return ctx;
  }

  private handleWorkletMessage(event: MessageEvent): void {
    const data = event.data as {
      pcm: ArrayBuffer;
      channels: number;
      sampleRate: number;
      frames: number;
    };
    if (!(data.pcm instanceof ArrayBuffer)) return;
    this.opts.bridge.sendAudioFrameDirect({
      windowId: this.opts.windowId,
      pcm: new Uint8Array(data.pcm),
      sampleRate: data.sampleRate,
      channels: data.channels,
      timestampNs: BigInt(Date.now()) * 1_000_000n,
    });
  }
}
