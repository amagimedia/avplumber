import { describe, expect, it } from 'vitest';
import { ConfigService } from '../../../src/main/config/ConfigService';
import { ValidationError } from '../../../src/main/rest/errors';

const svc = new ConfigService();

const validBody = {
  id: 'win-1',
  url: 'https://example.com/',
  width: 1280,
  height: 720,
  fps: 30,
  audio: false,
};

describe('ConfigService.validateWindowConfig', () => {
  it('accepts a valid body', () => {
    const cfg = svc.validateWindowConfig(validBody);
    expect(cfg).toEqual(validBody);
  });

  it('defaults audio to false when omitted', () => {
    const { audio: _drop, ...noAudio } = validBody;
    expect(svc.validateWindowConfig(noAudio).audio).toBe(false);
  });

  it('rejects non-object body', () => {
    expect(() => svc.validateWindowConfig('nope')).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig(null)).toThrow(ValidationError);
  });

  it('rejects missing or non-string id', () => {
    expect(() => svc.validateWindowConfig({ ...validBody, id: '' })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, id: 42 })).toThrow(ValidationError);
  });

  it('rejects ids that violate the regex', () => {
    expect(() => svc.validateWindowConfig({ ...validBody, id: 'a b' })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, id: 'a/b' })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, id: 'x'.repeat(65) })).toThrow(
      ValidationError,
    );
  });

  it('rejects bad URLs and unsupported protocols', () => {
    expect(() => svc.validateWindowConfig({ ...validBody, url: 'not-a-url' })).toThrow(
      ValidationError,
    );
    expect(() => svc.validateWindowConfig({ ...validBody, url: 'javascript:alert(1)' })).toThrow(
      ValidationError,
    );
  });

  it('accepts http/https/file/data URLs', () => {
    for (const url of [
      'http://localhost/',
      'https://example.com/',
      'file:///tmp/fixture.html',
      'data:text/html,<p>hi</p>',
    ]) {
      expect(svc.validateWindowConfig({ ...validBody, url }).url).toBe(url);
    }
  });

  it('rejects out-of-range dimensions and fps', () => {
    expect(() => svc.validateWindowConfig({ ...validBody, width: 0 })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, width: 9999 })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, height: 1.5 })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, fps: 0 })).toThrow(ValidationError);
    expect(() => svc.validateWindowConfig({ ...validBody, fps: 1000 })).toThrow(ValidationError);
  });

  it('rejects non-boolean audio', () => {
    expect(() => svc.validateWindowConfig({ ...validBody, audio: 'yes' })).toThrow(
      ValidationError,
    );
  });
});

describe('ConfigService.validateId / validateUpdateUrl / validateShow', () => {
  it('validateId checks id', () => {
    expect(svc.validateId({ id: 'win-1' })).toEqual({ id: 'win-1' });
    expect(() => svc.validateId({ id: '' })).toThrow(ValidationError);
    expect(() => svc.validateId({})).toThrow(ValidationError);
  });

  it('validateUpdateUrl checks id + url', () => {
    expect(svc.validateUpdateUrl({ id: 'win-1', url: 'https://x/' })).toEqual({
      id: 'win-1',
      url: 'https://x/',
    });
    expect(() => svc.validateUpdateUrl({ id: 'win-1' })).toThrow(ValidationError);
    expect(() => svc.validateUpdateUrl({ id: 'win-1', url: 'oops' })).toThrow(ValidationError);
  });

  it('validateShow defaults show to true', () => {
    expect(svc.validateShow({ id: 'win-1' })).toEqual({ id: 'win-1', show: true });
    expect(svc.validateShow({ id: 'win-1', show: false })).toEqual({
      id: 'win-1',
      show: false,
    });
    expect(() => svc.validateShow({ id: 'win-1', show: 'yes' })).toThrow(ValidationError);
  });
});
