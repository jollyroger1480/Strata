import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { cancelQueued, planAction, queuedWhen, stepsDone, usePlan, useQueue, type Plan, type QueuedMessage } from '../lib/plan';
import { pushToast } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { Btn, Icon, IconBtn } from './ui';

/** The steps with their state: a number, a check once done, a soft glow on the one being worked on. Used in the chat and in the sidebar. */
export function PlanSteps({ plan }: { plan: Plan }) {
  return <ol className="plan__steps">
    {plan.steps.map((s, i) => <li key={s.id} className="plan__step" data-state={s.state}>
      <span className="plan__no" aria-hidden="true">{s.state === 'done' ? <Icon name="check" size={12}/> : s.state === 'failed' ? <Icon name="x" size={12}/> : i + 1}</span>
      <span className="plan__text">{s.text}{s.note ? <small>{s.note}</small> : null}<span className="sr-only"> ({tr(s.state === 'done' ? 'done' : s.state === 'active' ? 'working on it' : s.state === 'failed' ? 'failed' : 'waiting')})</span></span>
    </li>)}
  </ol>;
}

/** The /plan the assistant proposed, in the chat: Approve, Edit, Reject, or Start as goal. Nothing runs until the person decides. */
export function PlanCard({ chatId }: { chatId: string | null | undefined }) {
  const plan = usePlan(chatId);
  const [editing, setEditing] = useState(false), [draft, setDraft] = useState<string[]>([]), [busy, setBusy] = useState(false);
  useEffect(() => { setEditing(false); }, [chatId, plan?.id]);
  if (!chatId || !plan || plan.state === 'rejected') return null;
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); }
    catch (e) { pushToast('error', tr("The plan was not changed"), plainFailureText(e instanceof Error ? e.message : String(e)).message); }
    finally { setBusy(false); }
  };
  const proposed = plan.state === 'proposed';
  if (editing) return <section className="plan" aria-label={tr("Edit the plan")}>
    <div className="plan__head"><b>{tr("Edit the plan")}</b></div>
    <div className="plan__edit">
      {draft.map((text, i) => <div className="plan__row" key={i}>
        <span className="plan__no" aria-hidden="true">{i + 1}</span>
        <input className="st-input" value={text} aria-label={tr("Step {0}", [i + 1])} onChange={e => setDraft(d => d.map((t, j) => j === i ? e.target.value : t))}/>
        <IconBtn name="x" size={15} title={tr("Remove this step")} disabled={draft.length < 2} onClick={() => setDraft(d => d.filter((_, j) => j !== i))}/>
      </div>)}
      <Btn size="sm" icon="plus" onClick={() => setDraft(d => [...d, ''])}>{tr("Add a step")}</Btn>
    </div>
    <div className="plan__acts">
      <Btn size="sm" variant="primary" disabled={busy || !draft.some(t => t.trim())} onClick={() => void run(async () => { await planAction(chatId, 'edit', draft.map(t => t.trim()).filter(Boolean)); setEditing(false); })}>{tr("Save plan")}</Btn>
      <Btn size="sm" onClick={() => setEditing(false)}>{tr("Cancel")}</Btn>
    </div>
  </section>;
  return <section className="plan" data-state={plan.state} aria-label={tr("Plan")}>
    <div className="plan__head"><b>{tr("Plan")}</b><span>{plan.state === 'proposed' ? tr('{0} steps', [plan.steps.length]) : tr('{0} of {1} done', [stepsDone(plan), plan.steps.length])}</span></div>
    <PlanSteps plan={plan}/>
    {proposed ? <p className="plan__note">{tr("Nothing runs until you approve.")}</p> : null}
    {proposed ? <div className="plan__acts">
      <Btn size="sm" variant="primary" icon="check" disabled={busy} onClick={() => void run(() => planAction(chatId, 'approve'))}>{tr("Approve")}</Btn>
      <Btn size="sm" icon="pencil" disabled={busy} onClick={() => { setDraft(plan.steps.map(s => s.text)); setEditing(true); }}>{tr("Edit")}</Btn>
      <Btn size="sm" disabled={busy} onClick={() => void run(() => planAction(chatId, 'reject'))}>{tr("Reject")}</Btn>
      <span className="plan__sp"/>
      <Btn size="sm" icon="bolt" disabled={busy} onClick={() => void run(() => planAction(chatId, 'start-goal'))}>{tr("Start as goal")}</Btn>
    </div> : null}
  </section>;
}

/** The same steps in the sidebar, so they stay in view while the goal runs. */
export function PlanPanel({ chatId }: { chatId: string | null | undefined }) {
  const plan = usePlan(chatId);
  if (!plan) return <aside className="ts-panel plan-panel" id="plan-content" aria-label={tr("Plan")}><p className="ts-empty">{tr("There is no plan in this chat.")}</p></aside>;
  return <aside className="ts-panel plan-panel" id="plan-content" aria-label={tr("Plan")}>
    <div className="plan__head"><b>{tr("Plan")}</b><span>{tr('{0} of {1} done', [stepsDone(plan), plan.steps.length])}</span></div>
    {plan.goal ? <p className="plan__goal">{plan.goal}</p> : null}
    <PlanSteps plan={plan}/>
  </aside>;
}

/** Messages sent while the assistant works: shown in the chat as they are, with when they will go out, and a way to take them back. */
export function QueuedMessages({ chatId }: { chatId: string | null | undefined }) {
  const items = useQueue(chatId), plan = usePlan(chatId);
  if (!chatId || !items.length) return null;
  return <>{items.map((q: QueuedMessage) => {
    const when = queuedWhen(q, plan);
    return <div className="queued" key={q.id}>
      <div className="queued__bubble">{q.text}</div>
      <div className="queued__line"><span>{tr(when.id, when.args)}</span>
        <button type="button" className="queued__x" title={tr("Cancel this message")} aria-label={tr("Cancel this message")} onClick={() => void cancelQueued(chatId, q.id).catch(e => pushToast('error', tr("The message was not cancelled"), plainFailureText(e instanceof Error ? e.message : String(e)).message))}><Icon name="x" size={13}/></button></div>
    </div>;
  })}</>;
}
