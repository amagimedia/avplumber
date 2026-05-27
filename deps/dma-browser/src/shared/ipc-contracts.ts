/**
 * Wire types shared between the main process and the preload script.
 * Keep this file dependency-free so it can be imported from either side.
 */

export const AUDIO_FRAME_CHANNEL = 'dma-browser:audio-frame';

export interface AudioFrameMessage {
  readonly windowId: string;
  /** Interleaved float32 PCM as a raw byte view. */
  readonly pcm: Uint8Array;
  readonly sampleRate: number;
  readonly channels: number;
  readonly timestampNs: bigint;
}

/** Exposed to the renderer via contextBridge under window.dmaBrowser. */
export interface DmaBrowserBridge {
  sendAudioFrame(frame: AudioFrameMessage): void;
}
