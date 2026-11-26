const { spawnSync } = require('node:child_process');
if (process.platform === 'linux') {
  const cc = spawnSync(process.env.CXX || 'c++', ['-std=c++17', '-O2', '-fPIC', '-shared',
    '-I' + require('node-api-headers').include_dir, 'receiver.cc', '-o', 'receiver.node'],
    { cwd: __dirname, stdio: 'inherit' });
  if (cc.error) console.error(cc.error.message);
  process.exitCode = cc.status ?? 1;
}
