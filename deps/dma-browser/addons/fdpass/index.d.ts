export function sendFd(socketPath: string, fd: number): Promise<void>;
export function sendFdWithInfo(
  socketPath: string,
  fd: number,
  texInfoBuffer: Buffer,
): Promise<void>;
export function createServer(socketPath: string): boolean;
export function broadcastFd(socketPath: string, fd: number, texInfoBuffer: Buffer): Promise<void>;
export function closeServer(socketPath: string): void;
export function setServerLogger(socketPath: string, callback: (line: string) => void): void;
export function monotonicTimeNs(): bigint;
export function close(): void;
