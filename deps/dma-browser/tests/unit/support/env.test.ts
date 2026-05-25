import { describe, expect, it } from 'vitest';
import { envFlag, envInt, envList, envString } from '../../../src/main/support/env';

describe('envFlag', () => {
  it('returns default when unset or empty', () => {
    expect(envFlag({}, 'X', true)).toBe(true);
    expect(envFlag({ X: '' }, 'X', false)).toBe(false);
  });

  it.each([
    ['1', true],
    ['true', true],
    ['TRUE', true],
    ['yes', true],
    ['on', true],
    ['0', false],
    ['false', false],
    ['no', false],
    ['off', false],
  ])('parses %s correctly', (v, expected) => {
    expect(envFlag({ X: v }, 'X', !expected)).toBe(expected);
  });

  it('falls back to default on garbage', () => {
    expect(envFlag({ X: 'maybe' }, 'X', true)).toBe(true);
    expect(envFlag({ X: 'maybe' }, 'X', false)).toBe(false);
  });
});

describe('envInt', () => {
  it('returns default when unset or empty', () => {
    expect(envInt({}, 'X', 5, 0, 100)).toBe(5);
    expect(envInt({ X: '' }, 'X', 5, 0, 100)).toBe(5);
  });

  it('parses and clamps to [min,max]', () => {
    expect(envInt({ X: '42' }, 'X', 0, 0, 100)).toBe(42);
    expect(envInt({ X: '-9' }, 'X', 0, 0, 100)).toBe(0);
    expect(envInt({ X: '999' }, 'X', 0, 0, 100)).toBe(100);
  });

  it('returns default on non-numeric', () => {
    expect(envInt({ X: 'abc' }, 'X', 7, 0, 100)).toBe(7);
  });
});

describe('envString', () => {
  it('returns default when unset or empty', () => {
    expect(envString({}, 'X', 'fallback')).toBe('fallback');
    expect(envString({ X: '' }, 'X', 'fallback')).toBe('fallback');
  });

  it('returns the value verbatim otherwise', () => {
    expect(envString({ X: 'angle' }, 'X', 'native')).toBe('angle');
  });
});

describe('envList', () => {
  it('returns empty array when unset or empty', () => {
    expect(envList({}, 'X')).toEqual([]);
    expect(envList({ X: '' }, 'X')).toEqual([]);
  });

  it('splits on commas, trims, drops empties', () => {
    expect(envList({ X: 'a, b , ,c ' }, 'X')).toEqual(['a', 'b', 'c']);
  });
});
