import { ValidationError } from '../rest/errors';
import type { ShowPayload, UpdateUrlPayload, WindowConfig } from './WindowConfig';

const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
const MIN_DIMENSION = 16;
const MAX_DIMENSION = 7680;
const MIN_FPS = 1;
const MAX_FPS = 240;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function requireString(obj: Record<string, unknown>, field: string): string {
  const v = obj[field];
  if (typeof v !== 'string' || v.length === 0) {
    throw new ValidationError(`Field "${field}" must be a non-empty string`);
  }
  return v;
}

function requireFiniteInt(
  obj: Record<string, unknown>,
  field: string,
  min: number,
  max: number,
): number {
  const v = obj[field];
  if (typeof v !== 'number' || !Number.isFinite(v) || !Number.isInteger(v)) {
    throw new ValidationError(`Field "${field}" must be a finite integer`);
  }
  if (v < min || v > max) {
    throw new ValidationError(`Field "${field}" must be in [${min}, ${max}], got ${v}`);
  }
  return v;
}

function requireBoolean(obj: Record<string, unknown>, field: string, defaultValue: boolean): boolean {
  const v = obj[field];
  if (v === undefined) return defaultValue;
  if (typeof v !== 'boolean') {
    throw new ValidationError(`Field "${field}" must be a boolean`);
  }
  return v;
}

function requireUrl(obj: Record<string, unknown>, field: string): string {
  const raw = requireString(obj, field);
  try {
    const parsed = new URL(raw);
    if (!['http:', 'https:', 'file:', 'data:'].includes(parsed.protocol)) {
      throw new ValidationError(
        `Field "${field}" must use http/https/file/data protocol (got "${parsed.protocol}")`,
      );
    }
  } catch (err) {
    if (err instanceof ValidationError) throw err;
    throw new ValidationError(`Field "${field}" is not a valid URL: ${String(err)}`);
  }
  return raw;
}

export class ConfigService {
  public validateWindowConfig(input: unknown): WindowConfig {
    if (!isRecord(input)) {
      throw new ValidationError('Request body must be a JSON object');
    }
    const id = requireString(input, 'id');
    if (!ID_RE.test(id)) {
      throw new ValidationError(
        'Field "id" must match /^[A-Za-z0-9_-]{1,64}$/ (alphanumeric, dash, underscore; 1-64 chars)',
      );
    }
    const url = requireUrl(input, 'url');
    const width = requireFiniteInt(input, 'width', MIN_DIMENSION, MAX_DIMENSION);
    const height = requireFiniteInt(input, 'height', MIN_DIMENSION, MAX_DIMENSION);
    const fps = requireFiniteInt(input, 'fps', MIN_FPS, MAX_FPS);
    const audio = requireBoolean(input, 'audio', false);
    return { id, url, width, height, fps, audio };
  }

  public validateId(input: unknown): { id: string } {
    if (!isRecord(input)) {
      throw new ValidationError('Request body must be a JSON object');
    }
    const id = requireString(input, 'id');
    if (!ID_RE.test(id)) {
      throw new ValidationError('Field "id" must match /^[A-Za-z0-9_-]{1,64}$/');
    }
    return { id };
  }

  public validateUpdateUrl(input: unknown): UpdateUrlPayload {
    const { id } = this.validateId(input);
    const obj = input as Record<string, unknown>;
    const url = requireUrl(obj, 'url');
    return { id, url };
  }

  public validateShow(input: unknown): ShowPayload {
    const { id } = this.validateId(input);
    const obj = input as Record<string, unknown>;
    const show = requireBoolean(obj, 'show', true);
    return { id, show };
  }
}
