import { tr } from '../lib/i18n';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Icon, type IconName } from './ui';
import { contextTokens } from '../lib/experience';
import { ContextRing } from './ContextRing';
import { Popover } from './Popover';
import { chatMetrics, compactTokens, elapsed } from '../lib/chat-metrics';
import { useStore } from '../lib/store';
import { useNative } from '../lib/native';
import type { Session } from '../lib/types';
function Chip({ icon, visual, label, title, children }: {
    icon?: IconName;
    visual?: ReactNode;
    label: string;
    title: string;
    children: ReactNode;
}) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLButtonElement>(null);
    return <><button ref={ref} className="telemetry__chip" aria-label={tr("{0} \u2014 show details", [title])} aria-description={tr(label)} aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(v => !v)}>{tr(visual ?? (icon ? <Icon name={icon} size={14}/> : null))}{tr(label)}</button>{tr(open ? <Popover anchor={ref.current} title={tr(title)} onClose={() => setOpen(false)} above>{tr(children)}</Popover> : null)}</>;
}
function Row({ label, value }: {
    label: string;
    value: ReactNode;
}) { return <div className="telepop__row"><span className="telepop__label">{tr(label)}</span><span className="telepop__value num">{tr(value)}</span></div>; }
/** Refresh once a second; smooth live changes and show exact final measurements. */
function useCalmSpeed(value: number | null, sessionId: string | null, streaming: boolean) {
    const latest = useRef(value);
    latest.current = value;
    const [shown, setShown] = useState(value);
    useEffect(() => {
        setShown(latest.current);
        if (!streaming)
            return;
        const timer = setInterval(() => setShown(previous => {
            const next = latest.current;
            if (next === null || previous === null)
                return next;
            return Math.abs(next - previous) < 1 ? next : previous + (next - previous) * .35;
        }), 1000);
        return () => clearInterval(timer);
    }, [sessionId, streaming]);
    return shown;
}
export function ChatTelemetry({ session }: {
    session: Session | null;
}) {
    const monitor = useStore(s => s.monitor);
    const engine = useStore(s => s.engine);
    // The context shown is the one the loaded model runs with (the engine's own --max-context), not a catalog or settings default.
    const native = useNative();
    const running = native?.instances.find(i => i.id === native.defaultId && i.status === 'ready') ?? native?.instances.find(i => i.status === 'ready');
    const runningContext = running ? Number(running.metrics?.engine?.max_context) || running.context || undefined : undefined;
    const m = chatMetrics(session, monitor, runningContext ?? (engine ? contextTokens(engine.beginner.contextLength) : undefined));
    const speed = useCalmSpeed(m.speed, session?.id ?? null, m.streaming);
    if (!m.hasReplies)
        return null;
    return <div className="telemetry xs muted" aria-label={tr("Chat statistics")}>
    <Chip icon="gauge" label={tr("{0}{1} tok/s{2}", [m.estimated ? tr('≈ ') : tr(''), speed === null ? tr('—') : Math.round(speed), m.streaming ? tr('') : tr(' · last reply')])} title={tr("Session statistics")}>{tr(m.estimated ? <p className="xs muted">{tr("Live estimate from response length and elapsed time. Final usage comes from the engine.")}</p> : null)}<Row label={tr("Thinking & writing")} value={elapsed(m.llmMs)}/><Row label={tr("Tool time")} value={elapsed(m.toolMs)}/><Row label={tr("Average time to first token")} value={m.ttft === null ? 'Not reported' : elapsed(m.ttft)}/><Row label={tr("Latest response speed")} value={speed === null ? 'Not reported' : `${Math.round(speed)} tokens/s`}/><Row label={tr("Turns / tool calls")} value={`${m.turns} / ${m.steps}`}/></Chip>
    <Chip icon="memory" label={tr("{0}{1} total tokens", [m.estimated ? tr('≈ ') : tr(''), compactTokens(m.total)])} title={tr("Token usage")}><Row label={tr("Total in this chat")} value={compactTokens(m.total)}/><Row label={tr("Input tokens")} value={compactTokens(m.input)}/><Row label={tr("Output tokens")} value={compactTokens(m.output)}/></Chip>
    <Chip visual={<ContextRing percent={m.pct}/>} label={tr(m.pct === null ? '—' : `${Math.round(m.pct)}% of context`)} title={tr("Context length")}><Row label={tr("Context used")} value={m.pct === null ? 'Not reported' : `${Math.round(m.pct)}%`}/><div className="context-meter" role="meter" aria-label={tr("Context window used")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={m.pct ?? undefined}><span style={{ width: `${m.pct ?? 0}%` }}/></div><Row label={tr("Current context")} value={compactTokens(m.used)}/><Row label={tr("Context limit")} value={compactTokens(m.max)}/><Row label={tr("Space remaining")} value={m.used !== null && m.max !== null ? compactTokens(Math.max(0, m.max - m.used)) : '—'}/></Chip>
  </div>;
}
