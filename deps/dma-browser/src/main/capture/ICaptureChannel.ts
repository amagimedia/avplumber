export interface ICaptureChannel {
  start(): Promise<void>;
  stop(): Promise<void>;
}

export interface CaptureStats {
  readonly paintCount: number;
  readonly droppedFrames: number;
  readonly droppedReasons: Readonly<Record<string, number>>;
  readonly txFrameCount: number;
  readonly releasedFrameCount: number;
  readonly retainedFrameCount: number;
  readonly lastPaintTsMs: number | null;
}
