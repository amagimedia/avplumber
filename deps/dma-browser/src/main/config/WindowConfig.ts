export interface WindowConfig {
  readonly id: string;
  readonly url: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  readonly audio: boolean;
}

export interface UpdateUrlPayload {
  readonly id: string;
  readonly url: string;
}

export interface ShowPayload {
  readonly id: string;
  readonly show: boolean;
}

export interface WindowSnapshot {
  readonly id: string;
  readonly url: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  readonly audio: boolean;
  readonly visible: boolean;
  readonly stats: WindowStats;
}

export interface WindowStats {
  readonly paintCount: number;
  readonly droppedFrames: number;
  readonly droppedReasons: Readonly<Record<string, number>>;
  readonly txFrameCount: number;
  readonly lastPaintTsMs: number | null;
}
