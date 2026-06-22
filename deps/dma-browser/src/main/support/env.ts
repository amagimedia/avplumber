export type Env = Readonly<Record<string, string | undefined>>;

export function envFlag(env: Env, name: string, defaultValue: boolean): boolean {
  const raw = env[name];
  if (raw === undefined || raw === '') return defaultValue;
  const v = raw.trim().toLowerCase();
  if (v === '1' || v === 'true' || v === 'yes' || v === 'on') return true;
  if (v === '0' || v === 'false' || v === 'no' || v === 'off') return false;
  return defaultValue;
}

export function envInt(
  env: Env,
  name: string,
  defaultValue: number,
  min: number,
  max: number,
): number {
  const raw = env[name];
  if (raw === undefined || raw === '') return defaultValue;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return defaultValue;
  return Math.max(min, Math.min(max, n));
}

export function envString(env: Env, name: string, defaultValue: string): string {
  const raw = env[name];
  if (raw === undefined || raw === '') return defaultValue;
  return raw;
}

export function envList(env: Env, name: string): readonly string[] {
  const raw = env[name];
  if (raw === undefined || raw === '') return [];
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
