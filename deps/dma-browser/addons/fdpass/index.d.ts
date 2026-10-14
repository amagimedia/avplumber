export function createServer(socketPath: string): boolean;
export interface SendResult {
  clients: number;
  sent: number;
  backpressure: number;
  disconnected: number;
  errors: number;
}
/**
 * Sends `fd` (the DMA-BUF plane) and, when `fenceFd` is a non-negative number, that sync_file fd
 * as the second fd of the same message. `texInfoBuffer` must flag the fence (TexInfoEncoder) exactly
 * when one is given, or the call rejects without sending. The caller keeps both fds.
 */
export function broadcastFd(
  socketPath: string,
  fd: number,
  texInfoBuffer: Buffer,
  fenceFd?: number,
): Promise<SendResult>;
export function closeServer(socketPath: string): bigint[];
export function setServerLogger(socketPath: string, callback: (line: string) => void): void;
export function setReleaseCallback(
  socketPath: string,
  callback: (frameNumber: bigint, reusable: boolean) => void,
): void;
export function monotonicTimeNs(): bigint;
