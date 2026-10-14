import { readFileSync, readdirSync } from 'node:fs';
import * as path from 'node:path';

// CPU use of this service's own process tree, read from /proc: the supervisor (or the single
// Electron process), every Electron worker and every Chromium child under them. The mixer
// container cannot see these processes, so GET /metrics/cpu hands the figures over.
//
// Cost: one /proc/<pid>/stat per process (every pid, to find the tree), each command line once
// per process lifetime, and thread files only for the GPU processes, all synchronous in one
// pass. With 40 windows (about 70 processes, over 1,000 threads) reading every thread's stat on
// each call cost the supervisor a quarter of a core at one call a second.
//
// Linux only. Tick counts in /proc are in USER_HZ, which the kernel fixes at 100 for user space.

export const USER_HZ = 100;

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
  /** The CPU its main thread last ran on. */
  readonly cpu: number;
  /** Read for GPU processes only; empty for the rest. */
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

export interface GpuThreadRow extends GroupRow {
  readonly name: GpuThreadKind;
  /** For "other": its three busiest thread names (digits collapsed to N). */
  readonly top?: readonly { readonly name: string; readonly pct: number }[];
}

export interface CoreRow {
  readonly cpu: number;
  /**
   * An approximation: a GPU process thread's time is charged to the core it last ran on, any
   * other process's time to the core its main thread last ran on.
   */
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

/**
 * A GPU process thread by comm (the kernel keeps 15 characters). The main thread is the one whose
 * tid is the pid: Electron leaves it named after the executable ("electron"), Chrome names it
 * CrGpuMain.
 */
export function gpuThreadKind(comm: string, isMainThread = false): GpuThreadKind {
  if (isMainThread || comm === 'CrGpuMain') return 'main';
  if (comm.startsWith('VizCompositorTh')) return 'viz';
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
    };
  }
  const pctOf = (ticks: number): number => (Math.max(0, ticks) * 100) / hz / elapsedS;
  const prevProcesses = new Map(
    prev.processes.map((p): [string, ProcessSample] => [idKey(p.pid, p.start), p]),
  );

  const processes = new Map<string, Accumulator>();
  const gpu = new Map<string, Accumulator>();
  const gpuOther = new Map<string, Accumulator>();
  const perCoreTicks = new Map<number, number>();
  const charge = (cpu: number, ticks: number): void => {
    if (cpu >= 0) perCoreTicks.set(cpu, (perCoreTicks.get(cpu) ?? 0) + ticks);
  };
  let total = 0;

  for (const proc of cur.processes) {
    const before = prevProcesses.get(idKey(proc.pid, proc.start));
    const processTicks = Math.max(0, proc.ticks - (before?.ticks ?? 0));
    const processPct = pctOf(processTicks);
    total += processPct;
    add(processes, proc.kind, processPct);
    if (proc.kind !== 'gpu' || proc.threads.length === 0) {
      charge(proc.cpu, processTicks);
      continue;
    }
    const prevThreads = new Map(
      before?.threads.map((t): [string, number] => [idKey(t.tid, t.start), t.ticks]),
    );
    for (const thread of proc.threads) {
      const prevTicks = prevThreads.get(idKey(thread.tid, thread.start)) ?? 0;
      const ticks = Math.max(0, thread.ticks - prevTicks);
      const pct = pctOf(ticks);
      charge(thread.cpu, ticks);
      const kind = gpuThreadKind(thread.comm, thread.tid === proc.pid);
      add(gpu, kind, pct);
      if (kind === 'other') add(gpuOther, threadGroupName(thread.comm), pct);
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

/** The file system calls a sample makes; tests count them. Null: the entry is gone. */
export interface ProcReader {
  list(dir: string): string[] | null;
  read(file: string): string | null;
}

export const fsReader: ProcReader = {
  list: (dir) => {
    try {
      return readdirSync(dir);
    } catch {
      return null;
    }
  },
  read: (file) => {
    try {
      return readFileSync(file, 'utf8');
    } catch {
      return null; // the process or thread exited meanwhile
    }
  },
};

const parseOrNull = (text: string | null): ReturnType<typeof parseStat> | null => {
  if (text === null) return null;
  try {
    return parseStat(text);
  } catch {
    return null; // torn or unexpected line: leave it out of this sample
  }
};

/**
 * Reads `rootPid` and every descendant from a /proc tree: one stat per process, the command line
 * of a process whose type `kindCache` (keyed by pid and start time) lacks, and the threads of
 * GPU processes only.
 */
export function readCpuSample(
  procRoot: string,
  rootPid: number,
  kindCache = new Map<string, ProcessKind>(),
  now: () => number = Date.now,
  reader: ProcReader = fsReader,
): CpuSample {
  const names = reader.list(procRoot);
  if (names === null) throw new Error(`${procRoot} is not readable`);
  const stats = new Map<number, ReturnType<typeof parseStat>>();
  for (const name of names) {
    if (!/^\d+$/.test(name)) continue;
    const stat = parseOrNull(reader.read(path.join(procRoot, name, 'stat')));
    if (stat) stats.set(Number(name), stat);
  }
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
  const processes: ProcessSample[] = [];
  for (const pid of tree) {
    const stat = stats.get(pid);
    if (!stat) continue;
    const key = idKey(pid, stat.start);
    live.add(key);
    let kind = kindCache.get(key);
    if (kind === undefined) {
      const raw = reader.read(path.join(procRoot, String(pid), 'cmdline'));
      if (raw === null) continue;
      // A zygote child rewrites its title into one string: split on spaces as well.
      kind = processKind(raw.split(/[\0 ]+/).filter((arg) => arg.length > 0));
      kindCache.set(key, kind);
    }
    const threads: ThreadSample[] = [];
    if (kind === 'gpu') {
      const taskDir = path.join(procRoot, String(pid), 'task');
      for (const tid of reader.list(taskDir) ?? []) {
        const t = parseOrNull(reader.read(path.join(taskDir, tid, 'stat')));
        if (!t) continue;
        const { start, comm, ticks, cpu } = t;
        threads.push({ tid: Number(tid), start, comm, ticks, cpu });
      }
    }
    processes.push({ pid, start: stat.start, kind, ticks: stat.ticks, cpu: stat.cpu, threads });
  }
  for (const key of kindCache.keys()) if (!live.has(key)) kindCache.delete(key);
  const hostStat = reader.read(path.join(procRoot, 'stat'));
  return {
    atMs: now(),
    processes,
    hostCores: hostStat === null ? [] : parseHostCores(hostStat),
  };
}

export interface CpuMetricsSource {
  metrics(): Promise<CpuMetrics>;
}

/**
 * Figures between successive calls. A call sooner than `minIntervalMs` after the last sample
 * returns that sample's figures again, so a second viewer never shrinks the window to nothing.
 * A sample is one synchronous pass, so calls never overlap.
 */
export class ProcessCpuSampler implements CpuMetricsSource {
  private readonly procRoot: string;
  private readonly rootPid: number;
  private readonly minIntervalMs: number;
  private readonly now: () => number;
  private readonly kinds = new Map<string, ProcessKind>();
  private last: CpuSample | null = null;
  private result: CpuMetrics | null = null;

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
    try {
      return Promise.resolve(this.metricsSync());
    } catch (err) {
      return Promise.reject(err instanceof Error ? err : new Error(String(err)));
    }
  }

  private metricsSync(): CpuMetrics {
    if (this.result && this.last && this.now() - this.last.atMs < this.minIntervalMs) {
      return this.result;
    }
    const cur = readCpuSample(this.procRoot, this.rootPid, this.kinds, this.now);
    this.result = groupCpu(this.last, cur);
    this.last = cur;
    return this.result;
  }
}
