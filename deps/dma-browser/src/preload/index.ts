import { AudioCapturePipeline } from './AudioCapturePipeline';
import { IpcBridge } from './IpcBridge';
import { WebGLSyncInstaller } from './WebGLSyncInstaller';
import { LoopAnimationSync } from './LoopAnimationSync';

function readArgFlag(name: string, defaultValue = ''): string {
  const prefix = `--${name}=`;
  for (const arg of process.argv) {
    if (arg.startsWith(prefix)) return arg.slice(prefix.length);
  }
  return defaultValue;
}

const windowId = readArgFlag('window-id', 'unknown');
const audioEnabled = readArgFlag('audio-enabled', '0') === '1';

new WebGLSyncInstaller().install();

const loopEpoch = readArgFlag('loop-sync-epoch-ms');
if (loopEpoch !== '' && Number.isSafeInteger(Number(loopEpoch)) && Number(loopEpoch) >= 0) {
  new LoopAnimationSync(Number(loopEpoch)).install();
}

const bridge = new IpcBridge();
bridge.install();

if (audioEnabled) {
  new AudioCapturePipeline({ windowId, bridge }).install();
}
