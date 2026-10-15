import { tr } from '../lib/i18n';
/** Response defaults used for the next chat or API request. */
import { useEffect, useState } from 'react';
import { Btn, Dialog } from '../components/ui';
import { saveEngine, useStore } from '../lib/store';
export function SamplingDrawer({ open, onClose }: {
    open: boolean;
    onClose: () => void;
}) {
    const engine = useStore(s => s.engine);
    const modelId = useStore(s => s.library?.loaded);
    const [tab, setTab] = useState<'sampling' | 'advanced'>('sampling');
    const [local, setLocal] = useState<Record<string, number | boolean>>({});
    const [pending, setPending] = useState(false), [error, setError] = useState('');
    useEffect(() => { if (open) {
        setLocal({});
        setError('');
        setTab('sampling');
    } }, [open, modelId]);
    const value = (id: string) => local[id] ?? engine?.expert.find(p => p.id === id)?.value ?? 0;
    const dirty = Object.entries(local).some(([id, v]) => v !== engine?.expert.find(p => p.id === id)?.value);
    async function save() {
        if (!dirty || pending)
            return;
        setPending(true);
        setError('');
        try {
            await saveEngine({ expert: Object.entries(local).map(([id, value]) => ({ id, value })) }, true);
            onClose();
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    function reset() {
        setLocal(Object.fromEntries((engine?.expert ?? []).filter(p => p.id.startsWith('sample.')).map(p => [p.id, p.type === 'toggle' ? Boolean(p.recommended) : Number(p.recommended)])));
        setError('');
    }
    function slider(id: string, label: string, min: number, max: number, step: number, hint: string) {
        return <div className="sampling-field" key={id}>
      <div className="row row--between"><label htmlFor={`response-${id}`} className="strong">{tr(label)}</label><output htmlFor={`response-${id}`}>{tr(Number(value(id)).toFixed(step < 1 ? 2 : 0))}</output></div>
      <input id={`response-${id}`} type="range" className="st-slider" min={min} max={max} step={step} value={Number(value(id))} disabled={pending || !engine} aria-describedby={`response-hint-${id}`} onChange={e => setLocal(prev => ({ ...prev, [id]: Number(e.target.value) }))}/>
      <p className="small muted" id={`response-hint-${id}`}>{tr(hint)}</p>
    </div>;
    }
    return <Dialog open={open} title={tr("Response settings")} onClose={() => !pending && onClose()} actions={<><Btn disabled={pending || !engine} onClick={reset}>{tr("Reset defaults")}</Btn><Btn disabled={pending} onClick={onClose}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || !dirty} onClick={() => void save()}>{tr(pending ? 'Saving…' : 'Save changes')}</Btn></>}>
    <div className="editor-tabs" role="tablist" aria-label={tr("Response settings sections")}>
      {tr((['sampling', 'advanced'] as const).map(t => <button key={t} id={`response-tab-${t}`} role="tab" aria-selected={tab === t} aria-controls={`response-panel-${t}`} tabIndex={tab === t ? 0 : -1} onClick={() => setTab(t)} onKeyDown={e => { if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        const next = tab === 'sampling' ? 'advanced' : 'sampling';
        setTab(next);
        document.getElementById(`response-tab-${next}`)?.focus();
    } }}>{tr(t === 'sampling' ? 'Sampling' : 'Advanced')}</button>))}
    </div>
    <div className="settings-tab-panel" role="tabpanel" id={`response-panel-${tab}`} aria-labelledby={`response-tab-${tab}`}>
      {tr(tab === 'sampling' ? <div className="stack">
        {tr(slider('sample.temp', 'Temperature', 0, 2, .05, 'Lower is focused; higher is more varied.'))}
        {tr(slider('sample.topP', 'Top-p', 0, 1, .01, 'Limits sampling to the most likely vocabulary.'))}
        {tr(slider('sample.topK', 'Top-k', 0, 200, 1, 'Candidate tokens. Zero removes this limit.'))}
      </div> : <div className="stack">
        {tr(slider('sample.repeat', 'Repetition penalty', .5, 3, .01, 'Above 1 discourages repeated text.'))}
        <div className="sampling-number-grid">{tr([{ id: 'sample.seed', label: 'Seed', min: -1, max: 2147483647 }, { id: 'sample.maxTokens', label: 'Max output tokens', min: 0, max: 1048576 }].map(field => <label className="input-field" key={field.id}><span>{tr(field.label)}</span><input className="st-input" type="number" min={field.min} max={field.max} step={1} disabled={pending || !engine} value={Number(value(field.id))} onChange={e => setLocal(prev => ({ ...prev, [field.id]: Number(e.target.value) }))}/><small className="muted">{tr(field.id === 'sample.seed' ? '−1 chooses a random seed.' : '0 uses the engine default.')}</small></label>))}</div>
        <div className="response-projection"><div><strong>{tr("Speed projection")}</strong><p className="small muted">{tr("Experimental control vector. Requires a compatible engine with the vector loaded.")}</p></div><button className="st-toggle" role="switch" aria-label={tr("Speed projection")} aria-checked={Boolean(value('sample.speedProjection'))} disabled={pending || !engine} onClick={() => setLocal(prev => ({ ...prev, 'sample.speedProjection': !Boolean(value('sample.speedProjection')) }))}/></div>
      </div>)}
    </div>
    {tr(error ? <p className="inline-error" role="alert">{tr(error)}</p> : null)}
  </Dialog>;
}
