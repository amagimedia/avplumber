import { promises as fs } from 'node:fs';
import * as path from 'node:path';

// CPU use of this service's own process tree, read from /proc: the supervisor (or the single
// Electron process), every Electron worker and every Chromium child under them. The mixer
// container cannot see these processes, so GET /metrics/cpu hands the figures over.
//
// Linux only. Tick counts in /proc are in USER_HZ, which the kernel fixes at 100 for user space.

export const USER_HZ = 100;
/** Thread groups listed in `threads` before the rest become one "other" row. */
export const THREAD_GROUPS = 8;

export type ProcessKind = 'browser' | 'gpu' | 'renderer' | 'utility' | 'zygote' | 'other';
export type GpuThreadKind = 'viz' | 'main' | 'io' | 'other';

export interface ThreadSample {
  readonly tid: number;
  /** Start time in ticks since boot: a reused tid is another thread. */
  readonly start: number;
  readonly comm: string;
  /** utime + stime */
  readonly ticks: number;
  /** The CPU the thread last ran on. */
  readonly cpu: number;
}

export interface ProcessSample {
  readonly pid: number;
  readonly start: number;
  readonly kind: ProcessKind;
  /** utime + stime of the process, including threads that have exited. */
  readonly ticks: number;
  readonly threads: readonly ThreadSample[];
}

export interface CpuSample {
  readonly atMs: number;
  readonly processes: readonly ProcessSample[];
  /** Per host CPU [busy, total] ticks from /proc/stat, by CPU index; empty when unreadable. */
  readonly hostCores: readonly (readonly [number, number])[];
}

export interface GroupRow {
  readonly n: number;
  /** Sum, in percent of one core. */
  readonly pct: number;
  /** pct / n */
  readonly avgPct: number;
  /** The busiest member (process or thread). */
  readonly maxPct: number;
}

export interface ProcessRow extends GroupRow {
  readonly type: ProcessKind;
}

export interface NamedRow extends GroupRow {
  readonly name: string;
}

export interface GpuThreadRow extends GroupRow {
  readonly name: GpuThreadKind;
  /** For "other": its three busiest thread names (digits collapsed to N). */
  readonly top?: readonly { readonly name: string; readonly pct: number }[];
}

export interface CoreRow {
  readonly cpu: number;
  /** Chromium's CPU time charged to the core each thread last ran on: an approximation. */
  readonly chromiumPct: number;
  /** The whole host's use of this core, or null when /proc/stat could not be read. */
  readonly hostPct: number | null;
}

export interface CpuMetrics {
  /** Seconds the figures cover; null on the first call, which only takes the baseline. */
  readonly intervalS: number | null;
  readonly hz: number;
  /** Every process of the tree together, in percent of one core. */
  readonly totalPct: number | null;
  readonly cores: {
    /** Cores' worth used: totalPct / 100. */
    readonly used: number | null;
    readonly hostVcpus: number | null;
    readonly perCore: readonly CoreRow[];
  };
  /** By Chromium process type, the largest total first. */
  readonly processes: readonly ProcessRow[];
  /** Threads of the GPU process(es): viz compositor, GPU main, IO, the rest. */
  readonly gpuThreads: readonly GpuThreadRow[];
  /** Every thread of the tree by name (digits as N), the largest total first, then "other". */
  readonly threads: readonly NamedRow[];
}

const round1 = (value: number): number => Math.round(value * 10) / 10;
/** A pid or tid with its start time: the kernel reuses ids. */
const idKey = (id: number, start: number): string => `${String(id)}:${String(start)}`;

/** Chromium's --type= switch, or the executable for the process that has none. */
export function processKind(cmdline: readonly string[]): ProcessKind {
  const type = cmdline.find((arg) => arg.startsWith('--type='))?.slice('--type='.length);
  switch (type) {
    case undefined:
      break;
    case 'gpu-process':
      return 'gpu';
    case 'renderer':
      return 'renderer';
    case 'utility':
      return 'utility';
    case 'zygote':
      return 'zygote';
    default:
      return 'other'; // broker, crashpad-handler, ...
  }
  // No --type: the Electron browser process, unless it is the Node supervisor, a shell or the
  // crash handler.
  const exe = path.basename(cmdline[0] ?? '');
  if (exe === '' || /^(node|nodejs|bash|sh|dash)$/.test(exe) || exe.includes('crashpad')) {
    return 'other';
  }
  return 'browser';
}

/** comm of a GPU process thread; the kernel keeps 15 characters of a name. */
export function gpuThreadKind(comm: string): GpuThreadKind {
  if (comm.startsWith('VizCompositorTh')) return 'viz';
  if (comm === 'CrGpuMain') return 'main';
  if (comm.startsWith('Chrome_ChildIOT') || comm.startsWith('Chrome_IOThread')) return 'io';
  return 'other';
}

/** Digits become N, so ThreadPoolForeg and CompositorTileW1..9 group by kind. */
export function threadGroupName(comm: string): string {
  return comm.replace(/\d+/g, 'N');
}

interface Accumulator {
  n: number;
  pct: number;
  max: number;
}

function add(acc: Map<string, Accumulator>, key: string, pct: number): void {
  const row = acc.get(key) ?? { n: 0, pct: 0, max: 0 };
  row.n += 1;
  row.pct += pct;
  row.max = Math.max(row.max, pct);
  acc.set(key, row);
}

function groupRow(acc: Accumulator): GroupRow {
  return {
    n: acc.n,
    pct: round1(acc.pct),
    avgPct: round1(acc.n > 0 ? acc.pct / acc.n : 0),
    maxPct: round1(acc.max),
  };
}

const byTotal = <T extends Accumulator>(a: [string, T], b: [string, T]): number =>
  b[1].pct - a[1].pct || b[1].max - a[1].max || a[0].localeCompare(b[0]);

/** Rows sorted by total, at most `limit` and then the rest as one "other" row. */
export function topGroups(acc: Map<string, Accumulator>, limit: number): NamedRow[] {
  const ranked = [...acc.entries()].sort(byTotal);
  const rows: NamedRow[] = ranked
    .slice(0, limit)
    .map(([name, value]) => ({ name, ...groupRow(value) }));
  const rest = ranked.slice(limit).map(([, value]) => value);
  if (rest.length > 0) {
    rows.push({
      name: 'other',
      ...groupRow({
        n: rest.reduce((sum, value) => sum + value.n, 0),
        pct: rest.reduce((sum, value) => sum + value.pct, 0),
        max: Math.max(...rest.map((value) => value.max)),
      }),
    });
  }
  return rows;
}

const GPU_THREAD_ORDER: readonly GpuThreadKind[] = ['viz', 'main', 'io', 'other'];

/** The figures between two samples. A process or thread first seen in `cur` counts from zero. */
export function groupCpu(prev: CpuSample | null, cur: CpuSample, hz = USER_HZ): CpuMetrics {
  const hostVcpus = cur.hostCores.length > 0 ? cur.hostCores.length : null;
  const elapsedS = prev ? (cur.atMs - prev.atMs) / 1000 : 0;
  if (!prev || elapsedS <= 0) {
    return {
      intervalS: null,
      hz,
      totalPct: null,
      cores: { used: null, hostVcpus, perCore: [] },
      processes: [],
      gpuThreads: [],
      threads: [],
    };
  }
  const pctOf = (ticks: number): number => (Math.max(0, ticks) * 100) / hz / elapsedS;
  const prevProcesses = new Map(
    prev.processes.map((p): [string, ProcessSample] => [idKey(p.pid, p.start), p]),
  );

  const processes = new Map<string, Accumulator>();
  const gpu = new Map<string, Accumulator>();
  const gpuOther = new Map<string, Accumulator>();
  const threads = new Map<string, Accumulator>();
  const perCoreTicks = new Map<number, number>();
  let total = 0;

  for (const proc of cur.processes) {
    const before = prevProcesses.get(idKey(proc.pid, proc.start));
    const processPct = pctOf(proc.ticks - (before?.ticks ?? 0));
    total += processPct;
    add(processes, proc.kind, processPct);
    const prevThreads = new Map(
      before?.threads.map((t): [string, number] => [idKey(t.tid, t.start), t.ticks]),
    );
    for (const thread of proc.threads) {
      const ticks = Math.max(
        0,
        thread.ticks - (prevThreads.get(idKey(thread.tid, thread.start)) ?? 0),
      );
      const pct = pctOf(ticks);
      add(threads, threadGroupName(thread.comm), pct);
      if (thread.cpu >= 0) {
        perCoreTicks.set(thread.cpu, (perCoreTicks.get(thread.cpu) ?? 0) + ticks);
      }
      if (proc.kind === 'gpu') {
        const kind = gpuThreadKind(thread.comm);
        add(gpu, kind, pct);
        if (kind === 'other') add(gpuOther, threadGroupName(thread.comm), pct);
      }
    }
  }

  const perCore: CoreRow[] = [];
  const cores = Math.max(cur.hostCores.length - 1, ...perCoreTicks.keys()) + 1;
  for (let cpu = 0; cpu < cores; cpu += 1) {
    const now = cur.hostCores[cpu];
    const then = prev.hostCores[cpu];
    const totalTicks = now && then ? now[1] - then[1] : 0;
    perCore.push({
      cpu,
      chromiumPct: round1(Math.min(100, pctOf(perCoreTicks.get(cpu) ?? 0))),
      hostPct:
        now && then && totalTicks > 0 ? round1(((now[0] - then[0]) * 100) / totalTicks) : null,
    });
  }

  const gpuThreads: GpuThreadRow[] = [];
  for (const name of GPU_THREAD_ORDER) {
    const acc = gpu.get(name);
    if (!acc) continue;
    if (name === 'other') {
      const top = [...gpuOther.entries()]
        .sort(byTotal)
        .slice(0, 3)
        .map(([thread, value]) => ({ name: thread, pct: round1(value.pct) }));
      gpuThreads.push({ name, ...groupRow(acc), top });
    } else {
      gpuThreads.push({ name, ...groupRow(acc) });
    }
  }

  return {
    intervalS: Math.round(elapsedS * 100) / 100,
    hz,
    totalPct: round1(total),
    cores: { used: Math.round(total) / 100, hostVcpus, perCore },
    processes: [...processes.entries()]
      .sort(byTotal)
      .map(([type, acc]) => ({ type: type as ProcessKind, ...groupRow(acc) })),
    gpuThreads,
    threads: topGroups(threads, THREAD_GROUPS),
  };
}

/** One /proc/<pid>/stat or task stat line: comm and the fields after it, numbered as in proc(5). */
export function parseStat(text: string): {
  comm: string;
  ppid: number;
  ticks: number;
  start: number;
  cpu: number;
} {
  const close = text.lastIndexOf(')'); // comm may hold spaces and parentheses
  const comm = text.slice(text.indexOf('(') + 1, close);
  const fields = text.slice(close + 1).trim().split(/\s+/);
  // fields[0] is field 3 (state): field N is fields[N - 3].
  const num = (field: number): number => Number(fields[field - 3]);
  const values = { comm, ppid: num(4), ticks: num(14) + num(15), start: num(22), cpu: num(39) };
  if (![values.ppid, values.ticks, values.start].every(Number.isFinite)) {
    throw new Error('malformed stat line');
  }
  return { ...values, cpu: Number.isFinite(values.cpu) ? values.cpu : -1 };
}

/** Per CPU [busy, total] ticks from /proc/stat; busy excludes idle and iowait. */
export function parseHostCores(text: string): [number, number][] {
  const cores: [number, number][] = [];
  for (const line of text.split('\n')) {
    const match = /^cpu(\d+)\s+(.*)$/.exec(line);
    if (!match?.[1] || match[2] === undefined) continue;
    // user nice system idle iowait irq softirq steal; guest time is already inside user and nice.
    const v = match[2].trim().split(/\s+/).slice(0, 8).map(Number);
    const total = v.reduce((sum, x) => sum + x, 0);
    cores[Number(match[1])] = [total - (v[3] ?? 0) - (v[4] ?? 0), total];
  }
  return cores;
}

async function readText(file: string): Promise<string | null> {
  try {
    return await fs.readFile(file, 'utf8');
  } catch {
    return null; // the process or thread exited meanwhile
  }
}

/** Reads `rootPid` and every descendant from a /proc tree. */
export async function readCpuSample(
  procRoot: string,
  rootPid: number,
  cmdlineCache = new Map<string, string[]>(),
  now: () => number = Date.now,
): Promise<CpuSample> {
  const pids = (await fs.readdir(procRoot)).filter((name) => /^\d+$/.test(name));
  const stats = new Map<number, ReturnType<typeof parseStat>>();
  await Promise.all(
    pids.map(async (pid) => {
      const text = await readText(path.join(procRoot, pid, 'stat'));
      if (text === null) return;
      try {
        stats.set(Number(pid), parseStat(text));
      } catch {
        // torn or unexpected line: leave the process out of this sample
      }
    }),
  );
  const children = new Map<number, number[]>();
  for (const [pid, stat] of stats) {
    const list = children.get(stat.ppid) ?? [];
    list.push(pid);
    children.set(stat.ppid, list);
  }
  const tree: number[] = [];
  const queue = stats.has(rootPid) ? [rootPid] : [];
  const seen = new Set(queue);
  for (let pid = queue.shift(); pid !== undefined; pid = queue.shift()) {
    tree.push(pid);
    for (const child of children.get(pid) ?? []) {
      if (seen.has(child)) continue;
      seen.add(child);
      queue.push(child);
    }
  }

  const live = new Set<string>();
  const processes = await Promise.all(
    tree.map(async (pid): Promise<ProcessSample | null> => {
      const stat = stats.get(pid);
      if (!stat) return null;
      const key = idKey(pid, stat.start);
      live.add(key);
      let cmdline = cmdlineCache.get(key);
      if (!cmdline) {
        const raw = await readText(path.join(procRoot, String(pid), 'cmdline'));
        if (raw === null) return null;
        // A zygote child rewrites its title into one string: split on spaces as well.
        cmdline = raw.split(/[\0 ]+/).filter((arg) => arg.length > 0);
        cmdlineCache.set(key, cmdline);
      }
      const taskDir = path.join(procRoot, String(pid), 'task');
      let tids: string[];
      try {
        tids = await fs.readdir(taskDir);
      } catch {
        return null;
      }
      const threads = await Promise.all(
        tids.map(async (tid): Promise<ThreadSample | null> => {
          const text = await readText(path.join(taskDir, tid, 'stat'));
          if (text === null) return null;
          try {
            const t = parseStat(text);
            return { tid: Number(tid), start: t.start, comm: t.comm, ticks: t.ticks, cpu: t.cpu };
          } catch {
            return null;
          }
        }),
      );
      return {
        pid,
        start: stat.start,
        kind: processKind(cmdline),
        ticks: stat.ticks,
        threads: threads.filter((t): t is ThreadSample => t !== null),
      };
    }),
  );
  for (const key of cmdlineCache.keys()) if (!live.has(key)) cmdlineCache.delete(key);
  const hostStat = await readText(path.join(procRoot, 'stat'));
  return {
    atMs: now(),
    processes: processes.filter((p): p is ProcessSample => p !== null),
    hostCores: hostStat === null ? [] : parseHostCores(hostStat),
  };
}

export interface CpuMetricsSource {
  metrics(): Promise<CpuMetrics>;
}

/**
 * Figures between successive calls. A call sooner than `minIntervalMs` after the last sample
 * returns that sample's figures again, so a second viewer never shrinks the window to nothing;
 * concurrent calls share one read.
 */
export class ProcessCpuSampler implements CpuMetricsSource {
  private readonly procRoot: string;
  private readonly rootPid: number;
  private readonly minIntervalMs: number;
  private readonly now: () => number;
  private readonly cmdlines = new Map<string, string[]>();
  private last: CpuSample | null = null;
  private result: CpuMetrics | null = null;
  private pending: Promise<CpuMetrics> | null = null;

  constructor(
    procRoot = '/proc',
    rootPid: number = process.pid,
    minIntervalMs = 500,
    now: () => number = Date.now,
  ) {
    this.procRoot = procRoot;
    this.rootPid = rootPid;
    this.minIntervalMs = minIntervalMs;
    this.now = now;
  }

  public metrics(): Promise<CpuMetrics> {
    if (this.result && this.last && this.now() - this.last.atMs < this.minIntervalMs) {
      return Promise.resolve(this.result);
    }
    this.pending ??= this.sample().finally(() => {
      this.pending = null;
    });
    return this.pending;
  }

  private async sample(): Promise<CpuMetrics> {
    const cur = await readCpuSample(this.procRoot, this.rootPid, this.cmdlines, this.now);
    const result = groupCpu(this.last, cur);
    this.last = cur;
    this.result = result;
    return result;
  }
}
