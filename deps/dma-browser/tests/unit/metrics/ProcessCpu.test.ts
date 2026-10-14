import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  ProcessCpuSampler,
  groupCpu,
  gpuThreadKind,
  parseHostCores,
  parseStat,
  processKind,
  readCpuSample,
  threadGroupName,
  type CpuSample,
  type ProcessKind,
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
  start = 1,
): CpuSample['processes'][number] {
  return { pid, start, kind, ticks, threads };
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
  const base: CpuSample = {
    atMs: 0,
    hostCores: [
      [0, 0],
      [0, 0],
    ],
    processes: [
      proc(1, 'other', 0),
      proc(10, 'browser', 100),
      proc(20, 'gpu', 1000, [
        thread(20, 'CrGpuMain', 500, 0),
        thread(21, 'VizCompositorTh', 300, 1),
        thread(22, 'Chrome_ChildIOT', 100, 1),
        thread(23, 'ThreadPoolForeg', 50, 0),
        thread(24, 'CrShaderCache', 50, 0),
      ]),
      proc(30, 'renderer', 0, [thread(30, 'CrRendererMain', 0, 1)]),
      proc(31, 'renderer', 0, [thread(31, 'CrRendererMain', 0, 1)]),
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
        proc(20, 'gpu', 1160, [
          thread(20, 'CrGpuMain', 560, 0),
          thread(21, 'VizCompositorTh', 380, 1),
          thread(22, 'Chrome_ChildIOT', 110, 1),
          thread(23, 'ThreadPoolForeg', 54, 0),
          thread(24, 'CrShaderCache', 56, 0),
        ]),
        proc(30, 'renderer', 60, [thread(30, 'CrRendererMain', 60, 1)]),
        proc(31, 'renderer', 20, [thread(31, 'CrRendererMain', 20, 1)]),
        proc(40, 'utility', 10, [thread(40, 'AudioThread', 10, 0)]),
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
    // Thread time charged to the core each thread last ran on; the host's own use beside it.
    expect(m.cores.perCore).toEqual([
      { cpu: 0, chromiumPct: 40, hostPct: 50 }, // 60 + 4 + 6 + 10 ticks
      { cpu: 1, chromiumPct: 85, hostPct: 25 }, // 80 + 10 + 60 + 20 ticks
    ]);
    expect(m.threads[0]).toEqual({
      name: 'VizCompositorTh',
      n: 1,
      pct: 40,
      avgPct: 40,
      maxPct: 40,
    });
    expect(m.threads.find((row) => row.name === 'CrRendererMain')).toEqual({
      name: 'CrRendererMain',
      n: 2,
      pct: 40,
      avgPct: 20,
      maxPct: 30,
    });
  });

  it('treats a reused pid or tid as new and never reports negative time', () => {
    const cur: CpuSample = {
      atMs: 1000,
      hostCores: [],
      processes: [proc(10, 'browser', 5, [thread(10, 'CrBrowserMain', 5, 0, 77)], 77)],
    };
    const m = groupCpu(base, cur);
    expect(m.processes).toEqual([{ type: 'browser', n: 1, pct: 5, avgPct: 5, maxPct: 5 }]);
    expect(m.cores.perCore).toEqual([{ cpu: 0, chromiumPct: 5, hostPct: null }]);
    const shrunk = groupCpu(base, { ...base, atMs: 1000, processes: [proc(10, 'browser', 50)] });
    expect(shrunk.totalPct).toBe(0);
  });

  it('keeps the eight largest thread groups and sums the rest as other', () => {
    const names = Array.from({ length: 11 }, (_, i) => `T${String.fromCharCode(97 + i)}`);
    const before: CpuSample = {
      atMs: 0,
      hostCores: [],
      processes: [proc(1, 'browser', 0, names.map((n, i) => thread(i + 1, n, 0)))],
    };
    const after: CpuSample = {
      atMs: 1000,
      hostCores: [],
      processes: [proc(1, 'browser', 66, names.map((n, i) => thread(i + 1, n, i + 1)))],
    };
    const rows = groupCpu(before, after).threads;
    expect(rows.map((row) => row.name)).toEqual([
      'Tk',
      'Tj',
      'Ti',
      'Th',
      'Tg',
      'Tf',
      'Te',
      'Td',
      'other',
    ]);
    expect(rows[8]).toEqual({ name: 'other', n: 3, pct: 6, avgPct: 2, maxPct: 3 });
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

  it('reads the root and its descendants only, with types from the command lines', async () => {
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
      { pid: 999, ppid: 1, cmdline: ['/usr/bin/other'], ticks: 50, threads: [[999, 'x', 50, 0]] },
    ]);
    const sample = await readCpuSample(root, 100, new Map(), () => 42);
    expect(sample.atMs).toBe(42);
    expect(sample.hostCores).toEqual([[10, 100]]);
    expect(sample.processes.map((p) => [p.pid, p.kind, p.ticks]).sort()).toEqual([
      [100, 'other', 3],
      [200, 'browser', 7],
      [300, 'zygote', 0],
      [400, 'renderer', 9],
    ]);
    expect(sample.processes.find((p) => p.pid === 400)?.threads).toEqual([
      { tid: 400, start: 1, comm: 'CrRendererMain', ticks: 9, cpu: 0 },
    ]);
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
          threads: [[210, 'CrGpuMain', ticks, 0]],
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
});
