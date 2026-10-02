/**
 * Thin logging wrapper over ExtensionContext.log.
 *
 * - Respects the configured level.
 * - Prefixes every line with [Memory] so it is greppable in AiderDesk logs.
 * - Never receives raw conversation content or secrets: callers pass counts,
 *   ids, categories and short labels only.
 */
import type { ExtensionContext } from '@aiderdesk/extensions';
import type { LoggingConfig } from './config';

type Level = 'debug' | 'info' | 'warn' | 'error';

const ORDER: Record<Level, number> = { debug: 10, info: 20, warn: 30, error: 40 };

export class Logger {
  private ctx: ExtensionContext | null = null;
  private cfg: LoggingConfig = { enabled: true, level: 'info' };

  bind(ctx: ExtensionContext): void {
    this.ctx = ctx;
  }

  setConfig(cfg: LoggingConfig): void {
    this.cfg = cfg;
  }

  private allowed(level: Level): boolean {
    if (!this.cfg.enabled) return false;
    return ORDER[level] >= ORDER[this.cfg.level];
  }

  private emit(level: Level, message: string): void {
    if (!this.allowed(level)) return;
    if (!this.ctx) return;
    // ExtensionContext.log accepts 'info' | 'error' | 'warn' | 'debug'
    this.ctx.log(`[Memory] ${message}`, level);
  }

  debug(message: string): void {
    this.emit('debug', message);
  }
  info(message: string): void {
    this.emit('info', message);
  }
  warn(message: string): void {
    this.emit('warn', message);
  }
  error(message: string): void {
    this.emit('error', message);
  }
}

export const logger = new Logger();
