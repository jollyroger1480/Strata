import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { nativeAction, refreshNative, useNative, type NativeJob } from '../lib/native';
import { NativeSettings } from './NativeSettings';
import { Btn, Dialog, ErrorNotice, Select } from './ui';
import { NativeProgress } from './NativeModels';
export function NativeBenchmark({ open, onClose }: {
    open: boolean;
    onClose: () => void;
}) {
    const native = useNative();
    const [selected, setSelected] = useState(''), [error, setError] = useState(''), [jobId, setJobId] = useState(''), [report, setReport] = useState<Record<string, any> | null>(null);
    const instance = native?.instances.find(i => i.id === selected) ?? native?.instances.find(i => i.id === native.defaultId) ?? native?.instances[0];
    const job = native?.jobs.find(j => j.id === jobId), active = native?.jobs.some(j => j.state === 'running');
    useEffect(() => {
        if (open)
            void api.get<Record<string, any> | null>('/api/native/benchmark').then(setReport).catch(() => { });
    }, [open]);
    useEffect(() => {
        if (job?.state === 'done' && job.result)
            setReport(job.result);
        if (job?.state === 'error')
            setError(job.error ?? job.summary);
    }, [job?.state]);
    async function run() {
        setError('');
        try {
            const job = await nativeAction<NativeJob>('operations', { kind: 'benchmark', args: { instanceId: instance?.id } });
            setJobId(job.id);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    function download() {
        if (!report)
            return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = 'strata-benchmark.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    return <Dialog open={open} title={tr("Measure performance")} onClose={onClose} actions={<><Btn onClick={onClose}>{tr("Close")}</Btn>{tr(job?.state === 'running' ? <Btn onClick={() => void nativeAction(`operations/${job.id}/cancel`).catch(e => setError(e.message))}>{tr("Stop benchmark")}</Btn> : null)}{tr(report ? <Btn icon="download" onClick={download}>{tr("Save report")}</Btn> : null)}<Btn variant="primary" disabled={!instance || instance.status !== 'ready' || active} onClick={() => void run()}>{tr(active ? 'Measuring…' : 'Run short benchmark')}</Btn></>}><div className="stack"><p className="small muted">{tr("A fixed prompt with thinking off and up to 200 output tokens. Usually takes 10\u201360 seconds. Results are measured by your engine.")}</p>{tr((native?.instances.length ?? 0) > 1 ? <Select ariaLabel={tr("Benchmark model")} value={instance?.id ?? ''} options={(native?.instances ?? []).map(i => ({ value: i.id, label: i.label }))} onChange={setSelected}/> : null)}{tr(!instance ? <p>{tr("Load a native model first.")}</p> : null)}{tr(job?.state === 'running' ? <p role="status">{tr(job.summary)}</p> : null)}{tr(report ? <div className="native-benchmark"><div><span>{tr("Generating")}</span><strong>{tr(report.outputTps == null ? '—' : Number(report.outputTps).toFixed(1))} <small>{tr("tok/s")}</small></strong></div><div><span>{tr("Reading")}</span><strong>{tr(report.promptTps == null ? '—' : Number(report.promptTps).toFixed(0))} <small>{tr("tok/s")}</small></strong></div><p className="xs muted">{tr(report.model)} · {tr(new Date(report.at).toLocaleString(localeTag()))} · {tr(report.settings)}</p></div> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</div></Dialog>;
}
export function NativeOperations() {
    const native = useNative();
    const [repository, setRepository] = useState(''), [model, setModel] = useState(''), [error, setError] = useState(''), [pending, setPending] = useState(false), [benchmark, setBenchmark] = useState(false), [logs, setLogs] = useState<{
        at: string;
        instanceId: string;
        line: string;
    }[]>([]), [showLogs, setShowLogs] = useState(false), [confirm, setConfirm] = useState('');
    const installed = native?.installed?.models ?? [], selected = installed.find(i => i.model === model) ?? installed[0], active = native?.jobs.find(j => j.state === 'running');
    useEffect(() => { setRepository(native?.repo ?? ''); }, [native?.repo]);
    useEffect(() => {
        if (!showLogs)
            return;
        let current = true;
        const load = () => void api.get<{
            logs: typeof logs;
        }>('/api/native/logs').then(value => {
            if (current)
                setLogs(value.logs);
        }).catch(e => setError(e.message));
        load();
        const timer = setInterval(load, 2500);
        return () => { current = false; clearInterval(timer); };
    }, [showLogs]);
    async function connect() {
        setPending(true);
        setError('');
        try {
            await api.patch('/api/native', { repository });
            await refreshNative(true);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    async function operate(kind: string) {
        setError('');
        try {
            await nativeAction('operations', { kind, args: { model: selected?.model } });
            setConfirm('');
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    function exportLogs() { const url = URL.createObjectURL(new Blob([logs.map(l => `${l.at} ${l.instanceId}: ${l.line}`).join('\n')], { type: 'text/plain' })); const a = document.createElement('a'); a.href = url; a.download = 'strata-engine.log'; a.click(); URL.revokeObjectURL(url); }
    return <div className="stack stack--lg native-engine-settings"><section className="st-card"><div className="section-toolbar"><h2 className="h3">{tr("Engine installation")}</h2><span className="st-pill">{tr(native?.available ? 'Connected' : 'Setup needed')}</span></div><p className="small muted">{tr("Your local Strata installation handles optimized models, companion files and engine updates.")}</p><details open={!native?.available}><summary className="small">{tr("Installation folder")}</summary><label className="input-field"><span>{tr("Strata folder")}</span><input className="st-input" aria-label={tr("Strata installation folder")} placeholder={tr("Folder containing setup.py")} value={repository} onChange={e => setRepository(e.target.value)}/></label><div className="native-settings-actions"><Btn disabled={pending || !repository.trim() || !!active} onClick={() => void connect()}>{tr(pending ? 'Connecting…' : 'Connect installation')}</Btn><Btn disabled={!!active} onClick={() => void refreshNative(true)}>{tr("Check again")}</Btn><a className="st-btn st-btn--ghost" href="https://github.com/Niko1221/Strata" target="_blank" rel="noreferrer">{tr("Get Strata")}</a></div></details>{tr(native?.error ? <p className="small inline-error">{tr(native.error)}</p> : <p className="small muted">{tr(native?.summary)}</p>)}</section>
 {tr(installed.length ? <section className="st-card"><h2 className="h3">{tr("Model settings & tools")}</h2><Select ariaLabel={tr("Native settings model")} value={selected?.model ?? ''} options={installed.map(m => ({ value: m.model, label: m.model_name ?? m.model }))} onChange={setModel}/>{tr(selected ? <NativeSettings model={selected.model}/> : null)}</section> : <p className="small muted">{tr("Prepare a model in Models to configure its engine parameters and MCP tools.")}</p>)}
 <section className="st-card"><h2 className="h3">{tr("Performance")}</h2><p className="small muted">{tr("Measure speed with a short test. Expert calibration tries engine settings and saves the fastest measured configuration.")}</p><div className="native-settings-actions"><Btn icon="gauge" disabled={!native?.instances.some(i => i.status === 'ready') || !!active} onClick={() => setBenchmark(true)}>{tr("Measure speed")}</Btn><Btn disabled={!native?.capabilities.calibration || !selected?.ready || !!active} onClick={() => setConfirm('calibrate')}>{tr("Calibrate engine\u2026")}</Btn></div>{tr(!native?.capabilities.calibration ? <p className="xs muted">{tr("Calibration is available on NVIDIA hardware supported by the installed version. Short benchmarks work with any running native instance.")}</p> : null)}</section>
 <section className="st-card"><div className="section-toolbar"><h2 className="h3">{tr("Diagnostics")}</h2><Btn size="sm" onClick={() => setShowLogs(v => !v)}>{tr(showLogs ? 'Hide logs' : 'View engine logs')}</Btn></div><p className="small muted">{tr("Downloads, preparation and engine failures include their original messages here.")}</p>{tr(showLogs ? <><pre className="native-log" aria-label={tr("Engine log")}>{logs.length ? logs.map(l => `${l.at} · ${l.line}`).join('\n') : 'No engine messages yet.'}</pre><Btn size="sm" icon="download" onClick={exportLogs}>{tr("Save logs")}</Btn></> : null)}{tr(native?.jobs.filter(j => j.state === 'error').slice(-3).map(j => <ErrorNotice key={j.id} message={j.error ?? j.summary}/>))}</section>
 {tr(active ? <NativeProgress job={active}/> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}
 <NativeBenchmark open={benchmark} onClose={() => setBenchmark(false)}/><Dialog open={!!confirm} compact title={tr("Calibrate this computer?")} onClose={() => setConfirm('')} actions={<><Btn onClick={() => setConfirm('')}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!!active || (native?.instances.length ?? 0) > 0} onClick={() => void operate('calibrate')}>{tr("Start calibration")}</Btn></>}><p>{tr("The computer will be busy for about 5–15 minutes. Unload models first.")} {tr("Strata will calibrate {0} and save the fastest measured settings.", [selected?.model_name ?? tr("The selected model")])}</p></Dialog>
 </div>;
}
