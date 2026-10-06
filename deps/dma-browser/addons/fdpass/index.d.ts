export function createServer(socketPath: string): boolean;
export interface SendResult {
  clients: number;
  sent: number;
  backpressure: number;
  disconnected: number;
  errors: number;
}
export function broadcastFd(socketPath: string, fd: number, texInfoBuffer: Buffer): Promise<SendResult>;
export function closeServer(socketPath: string): bigint[];
export function setServerLogger(socketPath: string, callback: (line: string) => void): void;
export function setReleaseCallback(
  socketPath: string,
  callback: (frameNumber: bigint, reusable: boolean) => void,
): void;
export function monotonicTimeNs(): bigint;
