import * as fs from 'node:fs';
import * as net from 'node:net';
import type { IpcMainEvent } from 'electron';
import { ipcMain } from 'electron';
import type { LogSink } from '../support/Logger';
import { AUDIO_FRAME_CHANNEL, type AudioFrameMessage } from '../../shared/ipc-contracts';
import type { ICaptureChannel } from './ICaptureChannel';

const MAX_PENDING_BYTES = 8 * 1024 * 1024; // 8 MiB backpressure cap per client

export interface AudioCaptureChannelOptions {
  readonly windowId: string;
  readonly socketPath: string;
  readonly log: LogSink;
}

interface AudioClient {
  readonly socket: net.Socket;
  pending: number;
  dropped: number;
}

export class AudioCaptureChannel implements ICaptureChannel {
  private readonly opts: AudioCaptureChannelOptions;
  private server: net.Server | null = null;
  private readonly clients = new Set<AudioClient>();
  private readonly boundIpcHandler = (event: IpcMainEvent, payload: AudioFrameMessage) =>
    this.handleAudioFrame(event, payload);

  constructor(opts: AudioCaptureChannelOptions) {
    this.opts = opts;
  }

  public async start(): Promise<void> {
    this.unlinkSocketSafely();
    const srv = net.createServer((sock) => this.acceptClient(sock));
    srv.on('error', (err) => this.opts.log.write(`audio server error: ${err.message}`));
    await new Promise<void>((resolve, reject) => {
      const onError = (err: Error): void => {
        srv.removeListener('error', onError);
        reject(err);
      };
      srv.once('error', onError);
      srv.listen(this.opts.socketPath, () => {
        srv.removeListener('error', onError);
        this.opts.log.write(`audio server listening at ${this.opts.socketPath}`);
        resolve();
      });
    });
    this.server = srv;
    ipcMain.on(AUDIO_FRAME_CHANNEL, this.boundIpcHandler);
  }

  public async stop(): Promise<void> {
    ipcMain.removeListener(AUDIO_FRAME_CHANNEL, this.boundIpcHandler);
    for (const client of this.clients) {
      try {
        client.socket.destroy();
      } catch {
        // ignore
      }
    }
    this.clients.clear();
    const srv = this.server;
    this.server = null;
    if (srv) {
      await new Promise<void>((resolve) => {
        srv.close(() => resolve());
      });
    }
    this.unlinkSocketSafely();
  }

  public clientCount(): number {
    return this.clients.size;
  }

  private acceptClient(socket: net.Socket): void {
    try {
      socket.setNoDelay(true);
    } catch {
      // ignore
    }
    const client: AudioClient = { socket, pending: 0, dropped: 0 };
    this.clients.add(client);
    this.opts.log.write('audio client connected');
    socket.on('drain', () => {
      client.pending = socket.writableLength;
    });
    socket.on('close', () => {
      this.clients.delete(client);
      this.opts.log.write(`audio client disconnected (dropped=${client.dropped})`);
    });
    socket.on('error', () => {
      this.clients.delete(client);
    });
  }

  private handleAudioFrame(event: IpcMainEvent, payload: AudioFrameMessage): void {
    if (payload.windowId !== this.opts.windowId) return;
    const pcm = payload.pcm;
    if (!(pcm instanceof Uint8Array) || pcm.byteLength === 0) return;
    const buf = Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (const client of this.clients) {
      if (client.pending + buf.length > MAX_PENDING_BYTES) {
        client.dropped++;
        continue;
      }
      try {
        const ok = client.socket.write(buf);
        client.pending = client.socket.writableLength;
        if (!ok) {
          // backpressure — the drain handler will reset pending
        }
      } catch {
        client.dropped++;
      }
    }
    // event is unused but kept for symmetry with the IPC signature.
    void event;
  }

  private unlinkSocketSafely(): void {
    try {
      if (fs.existsSync(this.opts.socketPath)) fs.unlinkSync(this.opts.socketPath);
    } catch {
      // ignore
    }
  }
}
