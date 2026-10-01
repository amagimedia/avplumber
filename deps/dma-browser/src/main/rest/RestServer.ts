import type { Server } from 'node:http';
import express, { type Express, type NextFunction, type Request, type Response } from 'express';
import { ConfigService } from '../config/ConfigService';
import type { WindowControl } from '../WindowControl';
import { toErrorResponse } from './errors';

export interface RestServerOptions {
  readonly host: string;
  readonly port: number;
}

/** Express 4 ignores the promise a handler returns; pass its rejection to the error middleware. */
function route(
  handler: (req: Request, res: Response) => Promise<void>,
): (req: Request, res: Response, next: NextFunction) => void {
  return (req, res, next) => {
    handler(req, res).catch(next);
  };
}

export class RestServer {
  private readonly manager: WindowControl;
  private readonly cfg: ConfigService;
  private readonly opts: RestServerOptions;
  public readonly app: Express;
  private server: Server | null = null;

  constructor(
    manager: WindowControl,
    opts: RestServerOptions,
    cfg: ConfigService = new ConfigService(),
  ) {
    this.manager = manager;
    this.opts = opts;
    this.cfg = cfg;
    this.app = express();
    this.app.disable('x-powered-by');
    this.app.use(express.json({ limit: '1mb' }));
    this.registerRoutes();
    this.app.use(this.errorMiddleware);
  }

  public async listen(): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const srv = this.app.listen(this.opts.port, this.opts.host, () => resolve());
      srv.once('error', reject);
      this.server = srv;
    });
  }

  public async close(): Promise<void> {
    const srv = this.server;
    this.server = null;
    if (!srv) return;
    await new Promise<void>((resolve) => srv.close(() => resolve()));
  }

  private registerRoutes(): void {
    this.app.post(
      '/workers/recover',
      route(async (req, res) => {
        if (!this.manager.recover) {
          res.status(409).json({ error: 'Browser recovery requires the multiprocess supervisor' });
          return;
        }
        await this.manager.recover(this.cfg.validateIds(req.body));
        res.json({ ok: true });
      }),
    );

    this.app.post('/workers/plan', (req, res) => {
      const windows = (req.body as { windows?: unknown }).windows;
      if (!Number.isInteger(windows) || (windows as number) < 0) {
        res.status(400).json({ error: 'windows must be a non-negative integer' });
        return;
      }
      // Advisory: a single-process service has no workers to plan.
      this.manager.plan?.(windows as number);
      res.json({ ok: true });
    });

    this.app.post(
      '/window/open',
      route(async (req, res) => {
        const cfg = this.cfg.validateWindowConfig(req.body);
        const snap = await this.manager.open(cfg);
        res.status(200).json(snap);
      }),
    );

    this.app.post(
      '/window/close',
      route(async (req, res) => {
        const { id } = this.cfg.validateId(req.body);
        await this.manager.close(id);
        res.status(200).json({ ok: true, id });
      }),
    );

    this.app.get(
      '/window/close/all',
      route(async (_req, res) => {
        await this.manager.closeAll();
        res.status(200).json({ ok: true });
      }),
    );

    this.app.post(
      '/window/refresh',
      route(async (req, res) => {
        const { id } = this.cfg.validateId(req.body);
        const snap = await this.manager.refresh(id);
        res.status(200).json(snap);
      }),
    );

    this.app.post(
      '/window/update',
      route(async (req, res) => {
        const { id, url } = this.cfg.validateUpdateUrl(req.body);
        const snap = await this.manager.update(id, url);
        res.status(200).json(snap);
      }),
    );

    this.app.post(
      '/window/show',
      route(async (req, res) => {
        const { id, show } = this.cfg.validateShow(req.body);
        const snap = await this.manager.show(id, show);
        res.status(200).json(snap);
      }),
    );

    this.app.get(
      '/status',
      route(async (_req, res) => {
        res.status(200).json(await this.manager.status());
      }),
    );

    this.app.use((_req, res) => {
      res.status(404).json({ error: 'Not found', code: 'RouteNotFound' });
    });
  }

  private readonly errorMiddleware = (
    err: unknown,
    _req: Request,
    res: Response,
    _next: NextFunction,
  ): void => {
    const { status, body } = toErrorResponse(err);
    res.status(status).json(body);
  };
}
