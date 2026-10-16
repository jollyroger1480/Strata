import { useEffect, useSyncExternalStore } from 'react';
import { api } from './client';

/**
 * A chat's /plan and its queued messages as the host reports them (contract in notes-split-agent6.md and notes-split-ui5.md). Both arrive as
 * whole snapshots, in the chat stream as {type:'plan', plan} and {type:'queue', items}, and as events {type:'plan'|'queue', chatId, ...}.
 * A snapshot replaces what was there; null or an empty list clears it. The host owns both; every button asks it and draws the answer.
 */
export type PlanState = 'proposed' | 'approved' | 'running' | 'done' | 'rejected';
export type StepState = 'pending' | 'active' | 'done' | 'failed';
export interface PlanStep { id: string; text: string; note?: string; state: StepState }
export interface Plan { id: string; goal: string; state: PlanState; steps: PlanStep[] }
export interface QueuedMessage { id: string; text: string; after?: string }

const PLAN_STATES = ['proposed', 'approved', 'running', 'done', 'rejected'];
const STEP_STATES = ['pending', 'active', 'done', 'failed'];

export function readPlan(raw: unknown): Plan | null {
  if (!raw || typeof raw !== 'object') return null;
  const p = raw as Record<string, unknown>;
  if (typeof p.id !== 'string' || !p.id || !PLAN_STATES.includes(String(p.state)) || !Array.isArray(p.steps)) return null;
  const steps: PlanStep[] = [];
  p.steps.forEach((s, i) => {
    if (!s || typeof s !== 'object') return;
    const r = s as Record<string, unknown>;
    if (typeof r.text !== 'string' || !r.text.trim()) return;
    steps.push({
      id: typeof r.id === 'string' && r.id ? r.id : String(i + 1), text: r.text,
      state: STEP_STATES.includes(String(r.state)) ? r.state as StepState : 'pending',
      ...(typeof r.note === 'string' && r.note ? { note: r.note } : {}),
    });
  });
  if (!steps.length) return null;
  return { id: p.id, goal: typeof p.goal === 'string' ? p.goal : '', state: p.state as PlanState, steps };
}

export function readQueue(raw: unknown): QueuedMessage[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((q): QueuedMessage[] => {
    if (!q || typeof q !== 'object') return [];
    const r = q as Record<string, unknown>;
    return typeof r.id === 'string' && r.id && typeof r.text === 'string' && r.text
      ? [{ id: r.id, text: r.text, ...(typeof r.after === 'string' && r.after ? { after: r.after } : {}) }] : [];
  });
}

const plans = new Map<string, Plan>();
const queues = new Map<string, QueuedMessage[]>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach(fn => fn());
const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };

export function applyPlanFrame(chatId: string, raw: unknown) {
  const plan = readPlan(raw), before = plans.get(chatId);
  if (!plan) { if (plans.delete(chatId)) emit(); return; }
  if (before && JSON.stringify(before) === JSON.stringify(plan)) return;
  plans.set(chatId, plan);
  emit();
}

export function applyQueueFrame(chatId: string, raw: unknown) {
  const items = readQueue(raw), before = queues.get(chatId) ?? [];
  if (JSON.stringify(before) === JSON.stringify(items)) return;
  if (items.length) queues.set(chatId, items); else queues.delete(chatId);
  emit();
}

const NONE: QueuedMessage[] = [];
export const planOf = (chatId: string | null | undefined): Plan | null => (chatId ? plans.get(chatId) ?? null : null);
export const queueOf = (chatId: string | null | undefined): QueuedMessage[] => (chatId ? queues.get(chatId) ?? NONE : NONE);

export function usePlan(chatId: string | null | undefined): Plan | null {
  const plan = useSyncExternalStore(subscribe, () => planOf(chatId), () => null);
  useEffect(() => { if (chatId) void loadPlan(chatId); }, [chatId]);
  return plan;
}

export function useQueue(chatId: string | null | undefined): QueuedMessage[] {
  const items = useSyncExternalStore(subscribe, () => queueOf(chatId), () => NONE);
  useEffect(() => { if (chatId) void loadQueue(chatId); }, [chatId]);
  return items;
}

export async function loadPlan(chatId: string) {
  try { applyPlanFrame(chatId, (await api.get<{ plan: unknown }>(`/api/plan?chatId=${encodeURIComponent(chatId)}`)).plan); }
  catch { /* an older host has no plan route: no plan */ }
}

export async function loadQueue(chatId: string) {
  try { applyQueueFrame(chatId, (await api.get<{ items: unknown }>(`/api/chat/queue?chatId=${encodeURIComponent(chatId)}`)).items); }
  catch { /* an older host has no queue route: nothing queued */ }
}

/** Approve, edit (with the new step texts), reject, or start the plan as a goal. The host answers with the plan as it now is. */
export async function planAction(chatId: string, action: 'approve' | 'edit' | 'reject' | 'start-goal', steps?: string[]) {
  const result = await api.post<{ plan: unknown }>(`/api/plan/${action}`, { chatId, ...(steps ? { steps } : {}) });
  applyPlanFrame(chatId, result.plan);
}

/** Take a queued message back before it is sent. */
export async function cancelQueued(chatId: string, id: string) {
  const result = await api.post<{ items: unknown }>('/api/chat/queue/cancel', { chatId, id });
  applyQueueFrame(chatId, result.items);
}

/** The words under a queued message: when it will be sent. `after` is the step id it waits for, if the host names one. */
export function queuedWhen(item: QueuedMessage, plan: Plan | null): { id: string; args?: unknown[] } {
  const step = item.after && plan ? plan.steps.findIndex(s => s.id === item.after) : -1;
  return step >= 0 ? { id: 'Queued — sends when step {0} finishes', args: [step + 1] } : { id: 'Queued — sends when the assistant is done' };
}

export const stepsDone = (plan: Plan) => plan.steps.filter(s => s.state === 'done').length;
