/**
 * /loop: send the same message again and again, every few minutes, while this chat is open.
 *
 *   /loop 10m check whether the build passes     start (the message is sent now, then every 10 minutes)
 *   /loop stop                                    stop
 *   /loop                                         say whether a loop is running
 *
 * The loop lives in the app window: it waits for the chat to be idle, and it stops when the window closes.
 * For work that must go on with the app closed, a task on the Tasks screen is the right tool.
 */

export const LOOP_MIN_MS = 60_000;
export const LOOP_MAX_MS = 24 * 3_600_000;
export const LOOP_MAX_RUNS = 100;

export type LoopCommand =
  | { kind: 'start'; everyMs: number; prompt: string }
  | { kind: 'stop' }
  | { kind: 'status' }
  | { kind: 'help'; problem: string };

const UNITS: Record<string, number> = { s: 1000, sec: 1000, second: 1000, seconds: 1000, m: 60_000, min: 60_000, mins: 60_000, minute: 60_000, minutes: 60_000, h: 3_600_000, hr: 3_600_000, hour: 3_600_000, hours: 3_600_000 };

/** The words shown when a /loop line is not understood. */
export const LOOP_USAGE = 'Type /loop, how often, then what to do. For example: /loop 10m check whether the build passes. Use /loop stop to stop.';

/** null means the text is an ordinary message. */
export function parseLoop(text: string): LoopCommand | null {
  const match = text.trim().match(/^\/loop(?:\s+([\s\S]*))?$/i);
  if (!match) return null;
  const rest = (match[1] ?? '').trim();
  if (!rest) return { kind: 'status' };
  if (/^(stop|off|cancel|end)$/i.test(rest)) return { kind: 'stop' };
  const timed = rest.match(/^(?:every\s+)?(\d+(?:\.\d+)?)\s*([a-z]+)\s+([\s\S]+)$/i);
  const unit = timed ? UNITS[timed[2].toLowerCase()] : undefined;
  if (!timed || !unit) return { kind: 'help', problem: 'Say how often first, like 10m (minutes) or 2h (hours), then what to do.' };
  const everyMs = Math.round(Number(timed[1]) * unit);
  if (everyMs < LOOP_MIN_MS) return { kind: 'help', problem: 'The shortest time between runs is 1 minute, so the model has time to finish each one.' };
  if (everyMs > LOOP_MAX_MS) return { kind: 'help', problem: 'The longest time between runs is 24 hours. For longer waits, use a task on the Tasks screen.' };
  return { kind: 'start', everyMs, prompt: timed[3].trim() };
}

/** "every 10 minutes", "every hour". */
export function loopEvery(ms: number): string {
  if (ms % 3_600_000 === 0) { const h = ms / 3_600_000; return h === 1 ? 'every hour' : `every ${h} hours`; }
  const m = Math.round(ms / 60_000);
  return m === 1 ? 'every minute' : `every ${m} minutes`;
}
