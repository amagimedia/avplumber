const fs = require('node:fs');

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

function summarize(trace) {
  const gaps = trace.paints
    .slice(1)
    .map((paint, index) => Number(BigInt(paint.monoNs) - BigInt(trace.paints[index].monoNs)) / 1e6);
  // Include boundary silence: a stream dying halfway through must not pass on its earlier gaps.
  const maxGapMs = Math.max(
    0,
    ...gaps,
    (trace.paints[0]?.unixMs ?? trace.endAtMs) - trace.startAtMs,
    trace.endAtMs - (trace.paints.at(-1)?.unixMs ?? trace.startAtMs),
  );
  const residuals = trace.samples.flatMap((sample) =>
    (sample.loops ?? [])
      .filter((loop) => typeof loop.elapsedMs === 'number')
      .map((loop) => loop.elapsedMs - (sample.unixMs - trace.epochMs)),
  );
  return {
    paints: trace.paints.length,
    maxGapMs,
    p95GapMs: percentile(gaps, 0.95),
    medianAnimationResidualMs: percentile(residuals, 0.5),
    p95AbsAnimationResidualMs: percentile(residuals.map(Math.abs), 0.95),
  };
}

function compare(a, b, { maxPhaseMs = 10, maxGapFrames = 2.5 } = {}) {
  const failures = [];
  for (const trace of [a, b]) {
    if (
      !Number.isInteger(trace.fps) ||
      trace.fps < 1 ||
      trace.fps > 240 ||
      !Number.isFinite(trace.startAtMs) ||
      !Number.isFinite(trace.endAtMs) ||
      trace.endAtMs <= trace.startAtMs
    )
      throw new Error('Invalid probe rate or time interval');
    for (let index = 0; index < trace.paints.length; index++) {
      const paint = trace.paints[index];
      const previous = trace.paints[index - 1];
      if (
        !Number.isFinite(paint.unixMs) ||
        paint.unixMs < trace.startAtMs ||
        paint.unixMs >= trace.endAtMs ||
        (previous &&
          (paint.unixMs < previous.unixMs || BigInt(paint.monoNs) <= BigInt(previous.monoNs)))
      ) {
        throw new Error('Invalid or nonmonotonic paint timestamps');
      }
    }
  }
  for (const key of ['kind', 'fps', 'epochMs', 'phase', 'startAtMs', 'endAtMs']) {
    if (a[key] !== b[key]) failures.push(`different ${key}`);
  }
  if (a.kind !== 'browser-sync-probe-v1') failures.push('unknown probe format');
  const periodMs = 1000 / a.fps;
  const summaries = [summarize(a), summarize(b)];
  for (const [index, trace] of [a, b].entries()) {
    const summary = summaries[index];
    if (trace.errors.length) failures.push(`${trace.label}: ${trace.errors.join('; ')}`);
    if (summary.paints < 2) failures.push(`${trace.label}: insufficient paints`);
    const expected = ((trace.endAtMs - trace.startAtMs) * trace.fps) / 1000;
    if (summary.paints < expected * 0.9 || summary.paints > expected * 1.1) {
      failures.push(`${trace.label}: paint count differs from configured rate by more than 10%`);
    }
    if (summary.maxGapMs > maxGapFrames * periodMs)
      failures.push(`${trace.label}: paint delivery gap`);
    if (
      summary.p95AbsAnimationResidualMs === null ||
      summary.p95AbsAnimationResidualMs > periodMs
    ) {
      failures.push(`${trace.label}: animation not within one frame of epoch`);
    }
  }
  // Nearest wall-time delivery measures phase, not frame identity. Gap checks above prevent
  // an apparently good phase result from concealing missing frames or freezes.
  const differences = [];
  let index = 0;
  for (const paint of a.paints) {
    if (!b.paints.length) break;
    while (
      index + 1 < b.paints.length &&
      Math.abs(b.paints[index + 1].unixMs - paint.unixMs) <=
        Math.abs(b.paints[index].unixMs - paint.unixMs)
    )
      index++;
    differences.push(Math.abs(b.paints[index].unixMs - paint.unixMs));
  }
  const p95PaintDifferenceMs = percentile(differences, 0.95);
  if (p95PaintDifferenceMs === null || p95PaintDifferenceMs > maxPhaseMs)
    failures.push('paint delivery phase outside tolerance');
  return {
    pass: failures.length === 0,
    failures,
    replicas: summaries,
    p95PaintDifferenceMs,
    limits: { maxPhaseMs, maxGapFrames },
    scope:
      'Paint callbacks and sampled DOM animation time; not pixel identity or consumer/output continuity.',
  };
}

module.exports = { compare };
if (require.main === module) {
  try {
    const [fileA, fileB] = process.argv.slice(2);
    if (!fileA || !fileB) throw new Error('Usage: node compare-sync.cjs <a.json> <b.json>');
    const result = compare(
      JSON.parse(fs.readFileSync(fileA, 'utf8')),
      JSON.parse(fs.readFileSync(fileB, 'utf8')),
    );
    console.log(JSON.stringify(result, null, 2));
    process.exitCode = result.pass ? 0 : 1;
  } catch (error) {
    console.error(String(error));
    process.exitCode = 1;
  }
}
