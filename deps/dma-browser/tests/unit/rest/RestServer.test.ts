import { describe, expect, it, vi } from 'vitest';
import request from 'supertest';
import { RestServer } from '../../../src/main/rest/RestServer';
import { ConfigService } from '../../../src/main/config/ConfigService';
import {
  CapacityError,
  ConflictError,
  NotFoundError,
  ValidationError,
} from '../../../src/main/rest/errors';
import type { WindowManager } from '../../../src/main/WindowManager';
import type { WindowConfig, WindowSnapshot } from '../../../src/main/config/WindowConfig';

function snapshotFor(cfg: WindowConfig): WindowSnapshot {
  return {
    id: cfg.id,
    url: cfg.url,
    width: cfg.width,
    height: cfg.height,
    fps: cfg.fps,
    audio: cfg.audio,
    visible: false,
    stats: {
      paintCount: 0,
      droppedFrames: 0,
      droppedReasons: {},
      txFrameCount: 0,
      lastPaintTsMs: null,
    },
  };
}

function buildManager(overrides: Partial<WindowManager> = {}): WindowManager {
  const mgr = {
    open: vi.fn(),
    close: vi.fn(),
    closeAll: vi.fn().mockResolvedValue(undefined),
    refresh: vi.fn(),
    update: vi.fn(),
    show: vi.fn(),
    status: vi.fn().mockReturnValue({ windows: [], count: 0, maxWindows: 8 }),
    ...overrides,
  };
  return mgr as unknown as WindowManager;
}

function buildServer(mgr: WindowManager) {
  return new RestServer(mgr, { host: '127.0.0.1', port: 0 }, new ConfigService());
}

const validBody = {
  id: 'win-1',
  url: 'https://example.com/',
  width: 1280,
  height: 720,
  fps: 30,
  audio: false,
};

describe('POST /window/open', () => {
  it('200 with snapshot on success', async () => {
    const mgr = buildManager({ open: vi.fn().mockResolvedValue(snapshotFor(validBody)) });
    const res = await request(buildServer(mgr).app).post('/window/open').send(validBody);
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ id: 'win-1', url: 'https://example.com/' });
  });

  it('400 on validation failure', async () => {
    const mgr = buildManager();
    const res = await request(buildServer(mgr).app).post('/window/open').send({});
    expect(res.status).toBe(400);
    expect(res.body.code).toBe('ValidationError');
  });

  it('409 on duplicate id', async () => {
    const mgr = buildManager({ open: vi.fn().mockRejectedValue(new ConflictError('dup')) });
    const res = await request(buildServer(mgr).app).post('/window/open').send(validBody);
    expect(res.status).toBe(409);
    expect(res.body.code).toBe('ConflictError');
  });

  it('503 on capacity error', async () => {
    const mgr = buildManager({ open: vi.fn().mockRejectedValue(new CapacityError('full')) });
    const res = await request(buildServer(mgr).app).post('/window/open').send(validBody);
    expect(res.status).toBe(503);
    expect(res.body.code).toBe('CapacityError');
  });

  it('500 on unknown error', async () => {
    const mgr = buildManager({ open: vi.fn().mockRejectedValue(new Error('boom')) });
    const res = await request(buildServer(mgr).app).post('/window/open').send(validBody);
    expect(res.status).toBe(500);
    expect(res.body.code).toBe('InternalError');
  });
});

describe('POST /window/close', () => {
  it('200 on success', async () => {
    const mgr = buildManager({ close: vi.fn().mockResolvedValue(undefined) });
    const res = await request(buildServer(mgr).app)
      .post('/window/close')
      .send({ id: 'win-1' });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ ok: true, id: 'win-1' });
  });

  it('404 when id not found', async () => {
    const mgr = buildManager({ close: vi.fn().mockRejectedValue(new NotFoundError('gone')) });
    const res = await request(buildServer(mgr).app)
      .post('/window/close')
      .send({ id: 'win-1' });
    expect(res.status).toBe(404);
    expect(res.body.code).toBe('NotFoundError');
  });

  it('400 on missing id', async () => {
    const mgr = buildManager();
    const res = await request(buildServer(mgr).app).post('/window/close').send({});
    expect(res.status).toBe(400);
  });
});

describe('POST /window/refresh /update /show', () => {
  it('refresh returns 200 with snapshot', async () => {
    const mgr = buildManager({ refresh: vi.fn().mockReturnValue(snapshotFor(validBody)) });
    const res = await request(buildServer(mgr).app)
      .post('/window/refresh')
      .send({ id: 'win-1' });
    expect(res.status).toBe(200);
    expect(res.body.id).toBe('win-1');
  });

  it('update returns 200 and forwards url', async () => {
    const mgr = buildManager({ update: vi.fn().mockReturnValue(snapshotFor(validBody)) });
    const res = await request(buildServer(mgr).app)
      .post('/window/update')
      .send({ id: 'win-1', url: 'https://x/' });
    expect(res.status).toBe(200);
    expect(mgr.update).toHaveBeenCalledWith('win-1', 'https://x/');
  });

  it('show returns 200 and forwards show flag', async () => {
    const mgr = buildManager({ show: vi.fn().mockReturnValue(snapshotFor(validBody)) });
    const res = await request(buildServer(mgr).app)
      .post('/window/show')
      .send({ id: 'win-1', show: false });
    expect(res.status).toBe(200);
    expect(mgr.show).toHaveBeenCalledWith('win-1', false);
  });

  it('refresh maps validation error to 400', async () => {
    const mgr = buildManager({
      refresh: vi.fn().mockImplementation(() => {
        throw new ValidationError('bad');
      }),
    });
    const res = await request(buildServer(mgr).app).post('/window/refresh').send({ id: 'x' });
    // ValidationError thrown inside manager is propagated as 400 by the middleware.
    expect(res.status).toBe(400);
  });
});

describe('GET /window/close/all and /status', () => {
  it('closes all', async () => {
    const mgr = buildManager();
    const res = await request(buildServer(mgr).app).get('/window/close/all');
    expect(res.status).toBe(200);
    expect(mgr.closeAll).toHaveBeenCalled();
  });

  it('returns status', async () => {
    const mgr = buildManager();
    const res = await request(buildServer(mgr).app).get('/status');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ windows: [], count: 0, maxWindows: 8 });
  });
});

describe('unknown routes', () => {
  it('returns 404 with RouteNotFound', async () => {
    const mgr = buildManager();
    const res = await request(buildServer(mgr).app).get('/no/such/path');
    expect(res.status).toBe(404);
    expect(res.body.code).toBe('RouteNotFound');
  });
});
