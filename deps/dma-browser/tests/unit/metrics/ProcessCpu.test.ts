import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';
import { afterEach, describe, expect, it } from 'vitest';
import {
  ProcessCpuSampler,
  fsReader,
  groupCpu,
  gpuThreadKind,
  parseHostCores,
  parseStat,
  processKind,
  readCpuSample,
  threadGroupName,
  type CpuSample,
  type ProcessKind,
  type ProcReader,
  type ThreadSample,
} from '../../../src/main/metrics/ProcessCpu';

/** A /proc stat line with the fields read here: ppid, utime, stime, starttime, processor. */
function statLine(
  id: number,
  comm: string,
  o: { ppid?: number; utime?: number; stime?: number; start?: number; cpu?: number } = {},
): string {
  const fields = Array.from({ length: 50 }, () => '0');
  const set = (field: number, value: number | string): void => {
    fields[field - 1] = String(value);
  };
  set(1, id);
  set(2, `(${comm})`);
  set(3, 'S');
  set(4, o.ppid ?? 0);
  set(14, o.utime ?? 0);
  set(15, o.stime ?? 0);
  set(22, o.start ?? 1);
  set(39, o.cpu ?? 0);
  return `${fields.join(' ')}\n`;
}

function thread(tid: number, comm: string, ticks: number, cpu = 0, start = 1): ThreadSample {
  return { tid, start, comm, ticks, cpu };
}

function proc(
  pid: number,
  kind: ProcessKind,
  ticks: number,
  threads: ThreadSample[] = [],
  o: { start?: number; cpu?: number } = {},
): CpuSample['processes'][number] {
  return { pid, start: o.start ?? 1, kind, ticks, cpu: o.cpu ?? 0, threads };
}

describe('processKind', () => {
  it.each([
    [['/app/electron', '--type=gpu-process', '--foo'], 'gpu'],
    [['/app/electron', '--type=renderer'], 'renderer'],
    [['/app/electron', '--type=utility', '--utility-sub-type=audio.mojom.Audio'], 'utility'],
    [['/app/electron', '--type=zygote'], 'zygote'],
    [['/app/electron', '--type=broker'], 'other'],
    [['/app/node_modules/electron/dist/electron', '--no-sandbox', 'dist/main/index.js'], 'browser'],
    [['node', '/app/dist/main/supervisor/index.js'], 'other'],
    [['/usr/bin/bash', 'bin/run.sh'], 'other'],
    [['/app/chrome_crashpad_handler', '--monitor-self'], 'other'],
    [[], 'other'],
  ] as const)('%j is %s', (cmdline, kind) => {
    expect(processKind(cmdline)).toBe(kind);
  });
});

describe('gpuThreadKind and threadGroupName', () => {
  it('splits the GPU process into viz, main, IO and the rest', () => {
    expect(gpuThreadKind('VizCompositorTh')).toBe('viz');
    expect(gpuThreadKind('CrGpuMain')).toBe('main');
    expect(gpuThreadKind('Chrome_ChildIOT')).toBe('io');
    expect(gpuThreadKind('ThreadPoolForeg')).toBe('other');
  });

  it('takes the thread whose tid is the pid as GPU main, whatever Electron named it', () => {
    expect(gpuThreadKind('electron', true)).toBe('main');
    expect(gpuThreadKind('electron')).toBe('other');
  });

  it('collapses digits', () => {
    expect(threadGroupName('CompositorTileW3')).toBe('CompositorTileWN');
    expect(threadGroupName('Media12')).toBe('MediaN');
  });
});

describe('parseStat and parseHostCores', () => {
  it('reads a comm with spaces and parentheses and the fields after it', () => {
    const line = statLine(7, 'a (b) c', { ppid: 3, utime: 10, stime: 5, start: 99, cpu: 6 });
    expect(parseStat(line)).toEqual({ comm: 'a (b) c', ppid: 3, ticks: 15, start: 99, cpu: 6 });
  });

  it('rejects a torn line', () => {
    expect(() => parseStat('7 (x) S')).toThrow();
  });

  it('reads busy and total ticks per CPU', () => {
    const text = 'cpu  9 9 9 9 9 9 9 9\ncpu0 10 0 5 80 5 0 0 0 0 0\ncpu1 1 1 1 1 1 1 1 1\nintr 1\n';
    expect(parseHostCores(text)).toEqual([
      [15, 100],
      [6, 8],
    ]);
  });
});

describe('groupCpu', () => {
  const gpuThreads = (main: number, viz: number, io: number, pool: number, shader: number) => [
    thread(20, 'electron', main, 0), // Electron's GPU main thread keeps the executable's name
    thread(21, 'VizCompositorTh', viz, 1),
    thread(22, 'Chrome_ChildIOT', io, 1),
    thread(23, 'ThreadPoolForeg', pool, 0),
    thread(24, 'CrShaderCache', shader, 0),
  ];
  const base: CpuSample = {
    atMs: 0,
    hostCores: [
      [0, 0],
      [0, 0],
    ],
    processes: [
      proc(1, 'other', 0),
      proc(10, 'browser', 100),
      proc(20, 'gpu', 1000, gpuThreads(500, 300, 100, 50, 50)),
      proc(30, 'renderer', 0, [], { cpu: 1 }),
      proc(31, 'renderer', 0, [], { cpu: 1 }),
    ],
  };

  it('takes a baseline on the first call', () => {
    const first = groupCpu(null, base);
    expect(first.intervalS).toBeNull();
    expect(first.totalPct).toBeNull();
    expect(first.processes).toEqual([]);
    expect(first.cores.hostVcpus).toBe(2);
  });

  it('groups by process type and GPU thread, the largest total first', () => {
    // Two seconds later, in ticks of 1/100 s: browser +20, GPU +160, renderers +60 and +20,
    // and a utility process started meanwhile with 10.
    const cur: CpuSample = {
      atMs: 2000,
      hostCores: [
        [100, 200],
        [50, 200],
      ],
      processes: [
        proc(1, 'other', 0),
        proc(10, 'browser', 120),
        proc(20, 'gpu', 1160, gpuThreads(560, 380, 110, 54, 56)),
        proc(30, 'renderer', 60, [], { cpu: 1 }),
        proc(31, 'renderer', 20, [], { cpu: 1 }),
        proc(40, 'utility', 10),
      ],
    };
    const m = groupCpu(base, cur);
    expect(m.intervalS).toBe(2);
    expect(m.totalPct).toBe(135); // 270 ticks over 2 s
    expect(m.cores.used).toBe(1.35);
    expect(m.processes).toEqual([
      { type: 'gpu', n: 1, pct: 80, avgPct: 80, maxPct: 80 },
      { type: 'renderer', n: 2, pct: 40, avgPct: 20, maxPct: 30 },
      { type: 'browser', n: 1, pct: 10, avgPct: 10, maxPct: 10 },
      { type: 'utility', n: 1, pct: 5, avgPct: 5, maxPct: 5 },
      { type: 'other', n: 1, pct: 0, avgPct: 0, maxPct: 0 },
    ]);
    expect(m.gpuThreads).toEqual([
      { name: 'viz', n: 1, pct: 40, avgPct: 40, maxPct: 40 },
      { name: 'main', n: 1, pct: 30, avgPct: 30, maxPct: 30 },
      { name: 'io', n: 1, pct: 5, avgPct: 5, maxPct: 5 },
      {
        name: 'other',
        n: 2,
        pct: 5,
        avgPct: 2.5,
        maxPct: 3,
        top: [
          { name: 'CrShaderCache', pct: 3 },
          { name: 'ThreadPoolForeg', pct: 2 },
        ],
      },
    ]);
    // GPU threads are charged to the core each last ran on, other processes to their main
    // thread's core; the host's own use of each core beside it.
    expect(m.cores.perCore).toEqual([
      { cpu: 0, chromiumPct: 50, hostPct: 50 }, // GPU 60 + 4 + 6, browser 20, utility 10
      { cpu: 1, chromiumPct: 85, hostPct: 25 }, // GPU 80 + 10, renderers 60 + 20
    ]);
  });

  it('treats a reused pid or tid as new and never reports negative time', () => {
    const cur: CpuSample = {
      atMs: 1000,
      hostCores: [],
      processes: [proc(10, 'browser', 5, [], { start: 77 })],
    };
    const m = groupCpu(base, cur);
    expect(m.processes).toEqual([{ type: 'browser', n: 1, pct: 5, avgPct: 5, maxPct: 5 }]);
    expect(m.cores.perCore).toEqual([{ cpu: 0, chromiumPct: 5, hostPct: null }]);
    const shrunk = groupCpu(base, { ...base, atMs: 1000, processes: [proc(10, 'browser', 50)] });
    expect(shrunk.totalPct).toBe(0);
  });
});

describe('readCpuSample and ProcessCpuSampler', () => {
  let root = '';
  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
    root = '';
  });

  interface FakeProcess {
    pid: number;
    ppid: number;
    cmdline: string[];
    ticks: number;
    /** [tid, comm, ticks, cpu] */
    threads: [number, string, number, number][];
  }

  function fakeProc(
    processes: FakeProcess[],
    hostStat = 'cpu  1 1 1 1 1 1 1 1\ncpu0 10 0 0 90 0 0 0 0\n',
  ): void {
    writeFileSync(path.join(root, 'stat'), hostStat);
    for (const p of processes) {
      const dir = path.join(root, String(p.pid));
      mkdirSync(path.join(dir, 'task'), { recursive: true });
      writeFileSync(path.join(dir, 'stat'), statLine(p.pid, 'x', { ppid: p.ppid, utime: p.ticks }));
      writeFileSync(path.join(dir, 'cmdline'), `${p.cmdline.join('\0')}\0`);
      for (const [tid, comm, ticks, cpu] of p.threads) {
        mkdirSync(path.join(dir, 'task', String(tid)), { recursive: true });
        writeFileSync(
          path.join(dir, 'task', String(tid), 'stat'),
          statLine(tid, comm, { ppid: p.ppid, utime: ticks, cpu }),
        );
      }
    }
  }

  /** fsReader that counts its calls by kind of file. */
  function countingReader(): { reader: ProcReader; counts: Record<string, number> } {
    const counts: Record<string, number> = {};
    const bump = (kind: string): void => {
      counts[kind] = (counts[kind] ?? 0) + 1;
    };
    return {
      counts,
      reader: {
        list: (dir) => {
          bump(dir.endsWith('task') ? 'taskList' : 'procList');
          return fsReader.list(dir);
        },
        read: (file) => {
          bump(file.includes(`${path.sep}task${path.sep}`) ? 'threadStat' : path.basename(file));
          return fsReader.read(file);
        },
      },
    };
  }

  it('reads the root and its descendants, and threads of GPU processes only', () => {
    root = mkdtempSync(path.join(tmpdir(), 'proc-'));
    fakeProc([
      { pid: 100, ppid: 1, cmdline: ['node', 'sv.js'], ticks: 3, threads: [[100, 'node', 3, 0]] },
      {
        pid: 200,
        ppid: 100,
        cmdline: ['/e/electron', 'main.js'],
        ticks: 7,
        threads: [[200, 'CrBrowserMain', 7, 0]],
      },
      { pid: 300, ppid: 200, cmdline: ['/e/electron', '--type=zygote'], ticks: 0, threads: [] },
      // A zygote child's title is one string with spaces.
      {
        pid: 400,
        ppid: 300,
        cmdline: ['/e/electron --type=renderer --lang=en'],
        ticks: 9,
        threads: [[400, 'CrRendererMain', 9, 0]],
      },
      {
        pid: 500,
        ppid: 200,
        cmdline: ['/e/electron', '--type=gpu-process'],
        ticks: 11,
        threads: [
          [500, 'electron', 6, 2],
          [501, 'VizCompositorTh', 5, 3],
        ],
      },
      { pid: 999, ppid: 1, cmdline: ['/usr/bin/other'], ticks: 50, threads: [[999, 'x', 50, 0]] },
    ]);
    const { reader, counts } = countingReader();
    const kinds = new Map<string, ProcessKind>();
    const sample = readCpuSample(root, 100, kinds, () => 42, reader);
    expect(sample.atMs).toBe(42);
    expect(sample.hostCores).toEqual([[10, 100]]);
    expect(sample.processes.map((p) => [p.pid, p.kind, p.ticks, p.threads.length]).sort()).toEqual([
      [100, 'other', 3, 0],
      [200, 'browser', 7, 0],
      [300, 'zygote', 0, 0],
      [400, 'renderer', 9, 0],
      [500, 'gpu', 11, 2],
    ]);
    const gpuThreads = [...(sample.processes.find((p) => p.pid === 500)?.threads ?? [])];
    expect(gpuThreads.sort((a, b) => a.tid - b.tid)).toEqual([
      { tid: 500, start: 1, comm: 'electron', ticks: 6, cpu: 2 },
      { tid: 501, start: 1, comm: 'VizCompositorTh', ticks: 5, cpu: 3 },
    ]);
    // One stat per pid in /proc, a command line per process of the tree, one GPU task listing.
    expect(counts).toEqual({ procList: 1, stat: 7, cmdline: 5, taskList: 1, threadStat: 2 });
    // The next sample knows every process's type: no command line is read again.
    const again = countingReader();
    readCpuSample(root, 100, kinds, () => 43, again.reader);
    expect(again.counts.cmdline).toBeUndefined();
  });

  it('costs little on a show-sized tree: 70 processes, 1,500 threads, 5 GPU processes', () => {
    root = mkdtempSync(path.join(tmpdir(), 'proc-'));
    const processes: FakeProcess[] = [
      { pid: 2, ppid: 1, cmdline: ['node', 'sv.js'], ticks: 0, threads: [] },
    ];
    let tid = 10_000;
    const threadsOf = (count: number, comm: string): [number, string, number, number][] =>
      Array.from({ length: count }, (_, i) => [(tid += 1), `${comm}${String(i)}`, i, i % 16]);
    for (let w = 0; w < 5; w += 1) {
      const browser = 100 + w * 100;
      processes.push(
        { pid: browser, ppid: 2, cmdline: ['/e/electron'], ticks: 0, threads: threadsOf(60, 'B') },
        {
          pid: browser + 1,
          ppid: browser,
          cmdline: ['/e/electron', '--type=gpu-process'],
          ticks: 0,
          threads: threadsOf(30, 'G'),
        },
      );
      for (let r = 0; r < 11; r += 1) {
        processes.push({
          pid: browser + 10 + r,
          ppid: browser,
          cmdline: ['/e/electron', r < 8 ? '--type=renderer' : '--type=utility'],
          ticks: 0,
          threads: threadsOf(22, 'R'),
        });
      }
    }
    while (processes.length < 70) {
      const pid = 1000 + processes.length;
      processes.push({
        pid,
        ppid: 100,
        cmdline: ['/e/electron', '--type=utility'],
        ticks: 0,
        threads: threadsOf(12, 'U'),
      });
    }
    fakeProc(processes);
    const threads = processes.reduce((sum, p) => sum + p.threads.length, 0);
    expect(processes.length).toBe(70);
    expect(threads).toBeGreaterThanOrEqual(1500);

    const { reader, counts } = countingReader();
    const kinds = new Map<string, ProcessKind>();
    let prev = readCpuSample(root, 2, kinds, Date.now, reader);
    for (const key of Object.keys(counts)) delete counts[key];
    const calls = 20;
    const started = performance.now();
    for (let i = 0; i < calls; i += 1) {
      const cur = readCpuSample(root, 2, kinds, () => prev.atMs + 5000, reader);
      groupCpu(prev, cur);
      prev = cur;
    }
    const msPerCall = (performance.now() - started) / calls;
    console.log(
      `/metrics/cpu on ${String(processes.length)} processes, ${String(threads)} threads: ` +
        `${msPerCall.toFixed(2)} ms per call`,
    );
    // Per call: one /proc listing, 70 process stats, the 5 GPU processes' 150 thread stats.
    expect(counts).toEqual({
      procList: calls,
      stat: calls * 71, // 70 processes and /proc/stat
      taskList: calls * 5,
      threadStat: calls * 150,
    });
  });

  it('reports figures between calls and reuses them within the minimum interval', async () => {
    root = mkdtempSync(path.join(tmpdir(), 'proc-'));
    const now = { ms: 0 };
    const sampler = new ProcessCpuSampler(root, 200, 500, () => now.ms);
    const gpu = (ticks: number): void => {
      fakeProc([
        { pid: 200, ppid: 1, cmdline: ['/e/electron'], ticks: 0, threads: [] },
        {
          pid: 210,
          ppid: 200,
          cmdline: ['/e/electron', '--type=gpu-process'],
          ticks,
          threads: [[210, 'electron', ticks, 0]],
        },
      ]);
    };
    gpu(100);
    expect((await sampler.metrics()).intervalS).toBeNull();
    now.ms = 1000;
    gpu(150);
    const second = await sampler.metrics();
    expect(second.intervalS).toBe(1);
    expect(second.processes[0]).toEqual({ type: 'gpu', n: 1, pct: 50, avgPct: 50, maxPct: 50 });
    expect(second.gpuThreads).toEqual([{ name: 'main', n: 1, pct: 50, avgPct: 50, maxPct: 50 }]);
    now.ms = 1200;
    gpu(400);
    expect(await sampler.metrics()).toBe(second);
  });

  it('rejects where /proc cannot be read', async () => {
    const sampler = new ProcessCpuSampler(path.join(tmpdir(), 'no-such-proc-dir'));
    await expect(sampler.metrics()).rejects.toThrow(/not readable/);
  });
});
