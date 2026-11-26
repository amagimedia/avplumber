import { expect, it } from 'vitest';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

it.each(['bash', 'sh'])('selects the correct renderer or supervisor with %s', (shell) => {
  const root = mkdtempSync(join(tmpdir(), 'isolated-launch-'));
  try {
    mkdirSync(join(root, 'bin'));
    mkdirSync(join(root, 'node_modules/.bin'), { recursive: true });
    const launcher = join(root, 'bin/run.sh');
    copyFileSync('bin/run.sh', launcher);
    const fake = join(root, 'node_modules/.bin/electron');
    writeFileSync(fake, '#!/bin/sh\nprintf "%s\\n" "$@"\n');
    chmodSync(fake, 0o700);
    const run = (env: NodeJS.ProcessEnv, ...args: string[]) =>
      spawnSync(shell, [launcher, ...args], {
        env: {
          PATH: process.env.PATH,
          DMA_BROWSER_DISABLE_NVIDIA_RUNTIME: '1',
          DMA_BROWSER_NODE_BIN: fake,
          ...env,
        },
        encoding: 'utf8',
      });
    const entry = join(root, 'dist/main/index.js');
    const grouped = { DMA_BROWSER_MAX_WINDOWS: '09', DMA_BROWSER_WINDOWS_PER_PROCESS: '08' };
    expect(run(grouped).stdout.trim()).toBe(join(root, 'dist/main/supervisor/index.js'));
    expect(run({}, 'two words').stdout).toBe(`--no-sandbox\n${entry}\ntwo words\n`);
    const isolated = { DMA_BROWSER_ISOLATED_SCENE_URL: 'http://127.0.0.1:43123/' };
    expect(run(isolated).stdout.trim()).toBe(entry);
    expect(run(isolated, '--no-sandbox').status).toBe(1);
    for (const config of [grouped, { DMA_BROWSER_PROCESS_COUNT: '2' }]) {
      const rejected = run({ ...isolated, ...config });
      expect(rejected.status).toBe(1);
      expect(rejected.stderr).toContain('single browser process');
      expect(rejected.stdout).toBe('');
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
