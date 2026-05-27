import * as fs from 'node:fs';
import * as path from 'node:path';

export interface LogSink {
  write(line: string): void;
  close(): void;
}

class FileLogSink implements LogSink {
  private stream: fs.WriteStream | null;

  constructor(filePath: string) {
    this.stream = fs.createWriteStream(filePath, { flags: 'a' });
  }

  public write(line: string): void {
    if (!this.stream) return;
    try {
      this.stream.write(line.endsWith('\n') ? line : line + '\n');
    } catch {
      // Swallow — logging must never throw.
    }
  }

  public close(): void {
    const s = this.stream;
    this.stream = null;
    if (s) {
      try {
        s.end();
      } catch {
        // ignore
      }
    }
  }
}

class NullLogSink implements LogSink {
  public write(_line: string): void {
    // discard
  }
  public close(): void {
    // no-op
  }
}

/**
 * Per-window log streams plus a shared global sink. Stream files live under
 * `${dir}/${id}.log` (windows) and `${dir}/dma-browser.log` (global).
 */
export class Logger {
  private readonly dir: string;
  private readonly perWindow = new Map<string, LogSink>();
  private readonly globalSink: LogSink;

  constructor(dir: string) {
    this.dir = dir;
    try {
      fs.mkdirSync(this.dir, { recursive: true });
    } catch {
      // ignore — `forWindow` and global will fall back to NullLogSink
    }
    this.globalSink = this.tryFileSink(path.join(this.dir, 'dma-browser.log'));
  }

  public global(line: string): void {
    this.globalSink.write(this.stamp(line));
  }

  public forWindow(id: string): LogSink {
    const existing = this.perWindow.get(id);
    if (existing) return existing;
    const sink = this.tryFileSink(path.join(this.dir, `${id}.log`));
    const stamped: LogSink = {
      write: (line) => sink.write(this.stamp(line)),
      close: () => sink.close(),
    };
    this.perWindow.set(id, stamped);
    return stamped;
  }

  public closeWindow(id: string): void {
    const sink = this.perWindow.get(id);
    if (sink) {
      sink.close();
      this.perWindow.delete(id);
    }
  }

  public closeAll(): void {
    for (const sink of this.perWindow.values()) sink.close();
    this.perWindow.clear();
    this.globalSink.close();
  }

  private stamp(line: string): string {
    return `[${new Date().toISOString()}] ${line}`;
  }

  private tryFileSink(filePath: string): LogSink {
    try {
      return new FileLogSink(filePath);
    } catch {
      return new NullLogSink();
    }
  }
}
