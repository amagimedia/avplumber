export abstract class DomainError extends Error {
  public abstract readonly httpStatus: number;
  public abstract readonly code: string;
}

export class ValidationError extends DomainError {
  public readonly httpStatus = 400;
  public readonly code = 'ValidationError';
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class NotFoundError extends DomainError {
  public readonly httpStatus = 404;
  public readonly code = 'NotFoundError';
  constructor(message: string) {
    super(message);
    this.name = 'NotFoundError';
  }
}

export class ConflictError extends DomainError {
  public readonly httpStatus = 409;
  public readonly code = 'ConflictError';
  constructor(message: string) {
    super(message);
    this.name = 'ConflictError';
  }
}

export class CapacityError extends DomainError {
  public readonly httpStatus = 503;
  public readonly code = 'CapacityError';
  constructor(message: string) {
    super(message);
    this.name = 'CapacityError';
  }
}

export function isDomainError(err: unknown): err is DomainError {
  return err instanceof DomainError;
}

export interface ErrorResponseBody {
  readonly error: string;
  readonly code: string;
  readonly details?: string;
}

export function toErrorResponse(err: unknown): { status: number; body: ErrorResponseBody } {
  if (isDomainError(err)) {
    return {
      status: err.httpStatus,
      body: { error: err.message, code: err.code },
    };
  }
  const message = err instanceof Error ? err.message : String(err);
  return {
    status: 500,
    body: { error: 'Internal server error', code: 'InternalError', details: message },
  };
}
