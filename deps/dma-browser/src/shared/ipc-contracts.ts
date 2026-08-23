/**
 * Wire types shared between the main process and the preload script.
 * Keep this file dependency-free so it can be imported from either side.
 */

export const AUDIO_FRAME_CHANNEL = 'dma-browser:audio-frame';
export const TRACE_FRAME_CHANNEL = 'dma-browser:trace-frame';
export const TRACE_PAINTED_CHANNEL = 'dma-browser:trace-painted';

export interface AudioFrameMessage {
  readonly windowId: string;
  /** Interleaved float32 PCM as a raw byte view. */
  readonly pcm: Uint8Array;
  readonly sampleRate: number;
  readonly channels: number;
  readonly timestampNs: bigint;
}

export interface TraceFrameMessage {
  readonly windowId: string;
  readonly traceId: bigint;
  readonly sequence: bigint;
  readonly sourcePtsMs: bigint;
  readonly rendererReceivedNs: bigint;
  readonly rafNs: bigint;
  readonly domAppliedNs: bigint;
}

export interface TracePaintedMessage {
  readonly traceIdStr: string;
  readonly sequenceStr: string;
  readonly frameNumberStr: string;
  readonly paintNsStr: string;
  readonly dmabufSendNsStr: string;
}

/** Exposed to the renderer via contextBridge under window.dmaBrowser. */
export interface DmaBrowserBridge {
  sendAudioFrame(frame: AudioFrameMessage): void;
  monotonicTimeNs(): string;
  markTraceFrame(frame: {
    windowId: string;
    traceIdStr: string;
    sequenceStr: string;
    sourcePtsMsStr: string;
    rendererReceivedNsStr: string;
    rafNsStr: string;
    domAppliedNsStr: string;
  }): void;
  onTracePainted(callback: (message: TracePaintedMessage) => void): void;
}
