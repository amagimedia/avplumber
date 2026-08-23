import { contextBridge, ipcRenderer } from 'electron';
import {
  AUDIO_FRAME_CHANNEL,
  TRACE_FRAME_CHANNEL,
  TRACE_PAINTED_CHANNEL,
  type AudioFrameMessage,
  type TraceFrameMessage,
  type TracePaintedMessage,
} from '../shared/ipc-contracts';

/**
 * Exposed in the renderer as `window.dmaBrowser`.
 *
 * Note: contextBridge cannot transfer BigInt, so we serialise `timestampNs`
 * as a decimal string at the bridge boundary and recover the BigInt on the
 * main-process side.
 */
export interface WireAudioFrame {
  readonly windowId: string;
  readonly pcm: ArrayBuffer;
  readonly sampleRate: number;
  readonly channels: number;
  readonly timestampNsStr: string;
}

export interface WireTraceFrame {
  readonly windowId: string;
  readonly traceIdStr: string;
  readonly sequenceStr: string;
  readonly sourcePtsMsStr: string;
  readonly rendererReceivedNsStr: string;
  readonly rafNsStr: string;
  readonly domAppliedNsStr: string;
}

export class IpcBridge {
  public install(): void {
    contextBridge.exposeInMainWorld('dmaBrowser', {
      sendAudioFrame: (frame: WireAudioFrame): void => {
        ipcRenderer.send(AUDIO_FRAME_CHANNEL, this.hydrate(frame));
      },
      monotonicTimeNs: (): string => process.hrtime.bigint().toString(),
      markTraceFrame: (frame: WireTraceFrame): void => {
        ipcRenderer.send(TRACE_FRAME_CHANNEL, this.hydrateTrace(frame));
      },
      onTracePainted: (callback: (message: TracePaintedMessage) => void): void => {
        ipcRenderer.on(TRACE_PAINTED_CHANNEL, (_event, message: TracePaintedMessage) => {
          callback(message);
        });
      },
    });
  }

  /**
   * Used when the preload itself calls main directly (no need to cross the
   * contextBridge). The `AudioCapturePipeline` calls this from preload-land.
   */
  public sendAudioFrameDirect(frame: AudioFrameMessage): void {
    ipcRenderer.send(AUDIO_FRAME_CHANNEL, frame);
  }

  private hydrate(frame: WireAudioFrame): AudioFrameMessage {
    return {
      windowId: frame.windowId,
      pcm: new Uint8Array(frame.pcm),
      sampleRate: frame.sampleRate,
      channels: frame.channels,
      timestampNs: BigInt(frame.timestampNsStr),
    };
  }

  private hydrateTrace(frame: WireTraceFrame): TraceFrameMessage {
    return {
      windowId: frame.windowId,
      traceId: BigInt(frame.traceIdStr),
      sequence: BigInt(frame.sequenceStr),
      sourcePtsMs: BigInt(frame.sourcePtsMsStr),
      rendererReceivedNs: BigInt(frame.rendererReceivedNsStr),
      rafNs: BigInt(frame.rafNsStr),
      domAppliedNs: BigInt(frame.domAppliedNsStr),
    };
  }
}
