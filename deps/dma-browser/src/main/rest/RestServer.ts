import type { Server } from 'node:http';
import express, { type Express, type NextFunction, type Request, type Response } from 'express';
import { ConfigService } from '../config/ConfigService';
import type { WindowControl } from '../WindowControl';
import { toErrorResponse } from './errors';

export interface RestServerOptions {
  readonly host: string;
  readonly port: number;
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
    this.app.post('/window/open', (req, res, next) => {
      void (async () => {
        try {
          const cfg = this.cfg.validateWindowConfig(req.body);
          const snap = await this.manager.open(cfg);
          res.status(200).json(snap);
        } catch (err) {
          next(err);
        }
      })();
    });

    this.app.post('/window/close', (req, res, next) => {
      void (async () => {
        try {
          const { id } = this.cfg.validateId(req.body);
          await this.manager.close(id);
          res.status(200).json({ ok: true, id });
        } catch (err) {
          next(err);
        }
      })();
    });

    this.app.get('/window/close/all', (_req, res, next) => {
      void (async () => {
        try {
          await this.manager.closeAll();
          res.status(200).json({ ok: true });
        } catch (err) {
          next(err);
        }
      })();
    });

    this.app.post('/window/refresh', (req, res, next) => {
      void (async () => {
        try {
          const { id } = this.cfg.validateId(req.body);
          const snap = await this.manager.refresh(id);
          res.status(200).json(snap);
        } catch (err) {
          next(err);
        }
      })();
    });

    this.app.post('/window/update', (req, res, next) => {
      void (async () => {
        try {
          const { id, url } = this.cfg.validateUpdateUrl(req.body);
          const snap = await this.manager.update(id, url);
          res.status(200).json(snap);
        } catch (err) {
          next(err);
        }
      })();
    });

    this.app.post('/window/show', (req, res, next) => {
      void (async () => {
        try {
          const { id, show } = this.cfg.validateShow(req.body);
          const snap = await this.manager.show(id, show);
          res.status(200).json(snap);
        } catch (err) {
          next(err);
        }
      })();
    });

    this.app.get('/status', (_req, res, next) => {
      void (async () => {
        try {
          res.status(200).json(await this.manager.status());
        } catch (err) {
          next(err);
        }
      })();
    });

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
