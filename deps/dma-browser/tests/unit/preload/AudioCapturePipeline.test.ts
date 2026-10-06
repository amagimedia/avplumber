import { afterEach, describe, expect, it, vi } from 'vitest';
import { AudioCapturePipeline } from '../../../src/preload/AudioCapturePipeline';
import type { IpcBridge } from '../../../src/preload/IpcBridge';

// Minimal Web Audio graph: like the real API, connecting nodes of different contexts throws.
class FakeNode {
  public readonly inputs: FakeNode[] = [];
  constructor(public readonly context: FakeAudioContext) {}
  public connect(target: FakeNode): void {
    if (target.context !== this.context) throw new Error('InvalidAccessError');
    target.inputs.push(this);
  }
}

class FakeGainNode extends FakeNode {
  public readonly gain = { value: 0 };
}

class FakeAudioWorkletNode extends FakeNode {
  public static readonly created: FakeAudioWorkletNode[] = [];
  public readonly port: { onmessage: unknown } = { onmessage: null };
  constructor(context: FakeAudioContext) {
    super(context);
    FakeAudioWorkletNode.created.push(this);
  }
}

class FakeAudioContext {
  public static readonly created: FakeAudioContext[] = [];
  public readonly destination = new FakeNode(this);
  public readonly audioWorklet = { addModule: (_url: string) => Promise.resolve() };
  constructor() {
    FakeAudioContext.created.push(this);
  }
  public createMediaElementSource(_el: unknown): FakeNode {
    return new FakeNode(this);
  }
  public createGain(): FakeGainNode {
    return new FakeGainNode(this);
  }
}

describe('AudioCapturePipeline', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    FakeAudioContext.created.length = 0;
    FakeAudioWorkletNode.created.length = 0;
  });

  it('feeds every media element found at load into one collector', async () => {
    vi.stubGlobal('AudioContext', FakeAudioContext);
    vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode);
    vi.stubGlobal(
      'MutationObserver',
      class {
        public observe = vi.fn();
      },
    );
    vi.stubGlobal('document', {
      readyState: 'complete',
      documentElement: {},
      querySelectorAll: () => [{}, {}],
    });
    const bridge = { sendAudioFrameDirect: vi.fn() } as unknown as IpcBridge;

    new AudioCapturePipeline({ windowId: 'w', bridge }).install();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(FakeAudioContext.created).toHaveLength(1);
    expect(FakeAudioWorkletNode.created).toHaveLength(1);
    const masterGain = FakeAudioWorkletNode.created[0]?.inputs[0];
    expect(masterGain?.inputs).toHaveLength(2);
  });
});
