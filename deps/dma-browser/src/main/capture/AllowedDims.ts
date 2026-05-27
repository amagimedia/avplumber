export interface Dimension {
  readonly w: number;
  readonly h: number;
}

export class AllowedDims {
  private readonly dims: readonly Dimension[] | null;

  private constructor(dims: readonly Dimension[] | null) {
    this.dims = dims;
  }

  public static fromList(list: readonly string[]): AllowedDims {
    if (list.length === 0) return new AllowedDims(null);
    const parsed: Dimension[] = [];
    for (const raw of list) {
      const m = /^(\d+)x(\d+)$/i.exec(raw.trim());
      if (!m) continue;
      const w = Number.parseInt(m[1] ?? '', 10);
      const h = Number.parseInt(m[2] ?? '', 10);
      if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
        parsed.push({ w, h });
      }
    }
    return parsed.length > 0 ? new AllowedDims(parsed) : new AllowedDims(null);
  }

  public allows(w: number, h: number): boolean {
    if (this.dims === null) return true;
    return this.dims.some((d) => d.w === w && d.h === h);
  }

  public listForLogging(): string {
    if (this.dims === null) return 'any';
    return this.dims.map((d) => `${d.w}x${d.h}`).join(',');
  }
}
