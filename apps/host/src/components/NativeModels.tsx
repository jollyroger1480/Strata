import { tr } from '../lib/i18n';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Screen } from './layout';
import { Btn, Dialog, ErrorNotice, Icon, Select } from './ui';
import { installProgress, useDownloadEta } from '../lib/download-eta';
import { plainFailureText } from '../lib/plain-errors';
import { api } from '../lib/client';
import { go, setSettingsSection, selectRemoteModel, pushToast } from '../lib/store';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { NativeSettings } from './NativeSettings';
import { RunImageModel } from './RunImageModel';
import { useStore } from '../lib/store';
import { NativePresets } from './NativePresets';
import { NativeBenchmark } from './NativeOperations';
import { nativeCatalogModels, catalogVersionLabel } from '../lib/native-catalog';
export function NativeModels({ onOtherModels, onReady, onImport, onConnectEngine }: {
    onOtherModels: () => void;
    onImport?: () => void;
    onConnectEngine?: () => void;
    onReady?: () => void;
}) {
    const native = useNative(), imageLibrary = useStore(s => s.imageModels), hardware = useStore(s => s.hardware);
    const [family, setFamily] = useState(''), [size, setSize] = useState(''), [context, setContext] = useState(32768), [vision, setVision] = useState('no');
    const manual = useStore(s => s.setup?.choices?.onboarding?.manual ?? false);
    const [expert, setExpert] = useState(manual), [multiple, setMultiple] = useState(false), [gpus, setGpus] = useState<number[]>([]), [kv, setKv] = useState('int8'), [lowRam, setLowRam] = useState('auto'), [split, setSplit] = useState('auto'), [deviceChoice, setDeviceChoice] = useState('auto');
    const [resourceProfile, setResourceProfile] = useState<'work' | 'performance'>('work');
    const [pending, setPending] = useState(false), [error, setError] = useState(''), [plan, setPlan] = useState<Record<string, any> | null>(null), [benchmark, setBenchmark] = useState(false), [jobId, setJobId] = useState(''), [manage, setManage] = useState(false);
    const presetPending = useRef(false);
    const families = nativeCatalogModels(native?.catalog?.families ?? []);
    const expertSection = useRef<HTMLElement>(null);
    useEffect(() => { if (expert)
        expertSection.current?.scrollIntoView({ block: 'start', behavior: 'smooth' }); }, [expert]);
    const selectedFamily = families.find(f => f.family === family || f.sizes.some(size => size.sourceFamily === family)) ?? families[0], selectedSize = selectedFamily?.sizes.find(s => s.id === size) ?? selectedFamily?.sizes[0];
    const installed = native?.installed?.models.find(m => m.model === selectedSize?.id);
    // A model that is already on this PC loads as it was prepared: the page shows its settings, so loading it never asks to download.
    useEffect(() => { if (installed?.ready) {
        if (installed.context)
            setContext(installed.context);
        if (installed.kv)
            setKv(installed.kv as typeof kv);
        setLowRam((installed.lowRamMode ?? 'auto') as typeof lowRam);
    } }, [installed?.model, installed?.ready, installed?.context, installed?.kv, installed?.lowRamMode]);
    const active = native?.jobs.find(j => j.state === 'running');
    const job = native?.jobs.find(j => j.id === jobId) ?? (!jobId ? native?.jobs.filter(j => ['load', 'install'].includes(j.kind)).at(-1) : undefined);
    const noFit = selectedSize?.on_this_pc === 'does not fit';
    const values = useMemo(() => ({ family: selectedSize?.sourceFamily, model: selectedSize?.model, context, vision: selectedSize?.images ? vision : 'no', kv, low_ram: lowRam, resourceProfile, ...(expert && gpus.length ? { gpus: gpus.join(',') } : {}), loadAfter: true, multiple: expert && multiple }), [selectedSize?.sourceFamily, selectedSize?.model, context, vision, kv, lowRam, resourceProfile, gpus.join(','), expert, multiple]);
    const downloadValues = { ...values, vision: selectedSize?.images ? 'yes' : 'no', loadVision: selectedSize?.images ? vision : 'no' };
    useEffect(() => { if (!native?.catalog || family)
        return; const current = native.instances.find(i => i.id === native.defaultId); const rec = native.catalog.recommendation ?? { family: null, model: null }; const loadedFamily = families.find(f => f.sizes.some(s => s.id === current?.model)); const have = new Set((native.installed?.models ?? []).filter(m => m.ready).map(m => m.model)); const installedFamily = families.find(f => f.sizes.some(s => have.has(s.id))); const f = loadedFamily ?? installedFamily ?? families.find(f => f.family === rec.family || f.sizes.some(s => s.sourceFamily === rec.family)) ?? families[0]; if (f) {
        setFamily(f.family);
        setSize(f.sizes.find(s => s.id === current?.model)?.id ?? f.sizes.find(s => have.has(s.id))?.id ?? f.sizes.find(s => s.model === rec.model)?.id ?? f.sizes[0]?.id ?? '');
        setContext(('context' in rec ? rec.context : undefined) ?? 32768);
    } }, [native?.catalog]);
    useEffect(() => { if (presetPending.current) {
        presetPending.current = false;
        return;
    } if (installed) {
        setContext(installed.context ?? 32768);
        setVision(installed.visionEnabled === false || installed.images === 'off' ? 'no' : installed.visionDevice ?? installed.images ?? 'no');
        setKv(installed.kv ?? 'int8');
        setLowRam(installed.lowRamMode ?? 'auto');
        setResourceProfile(installed.resourceProfile ?? 'performance');
    } }, [installed?.model, installed?.context, installed?.kv, installed?.images, installed?.visionEnabled, installed?.visionDevice, installed?.lowRamMode, installed?.resourceProfile]);
    useEffect(() => { if (jobId && job?.state === 'done' && (job.kind === 'load' || job.kind === 'install')) {
        selectRemoteModel(null);
        pushToast('success', 'Model ready', 'Start a new chat when you’re ready.');
    } }, [job?.state]);
    async function perform(action: () => Promise<unknown>) { setPending(true); setError(''); try {
        await action();
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    async function load() { if (!selectedSize)
        return; await perform(async () => { if (installed?.ready && context === installed.context && (!selectedSize?.images || installed.images && installed.images !== 'off') && (!installed.kv || kv === installed.kv) && lowRam === (installed.lowRamMode ?? 'auto')) {
        const result = await nativeAction<NativeJob>('operations', { kind: 'load', args: { model: selectedSize.id, resourceProfile, vision: selectedSize?.images ? vision : 'no', multiple: expert && multiple, ...(expert && gpus.length ? { gpus, layerSplit: split } : {}), values: {} } });
        setJobId(result.id);
    }
    else
        setPlan(await api.post('/api/native/plan', downloadValues)); }); }
    async function download() { await perform(async () => { const result = await nativeAction<NativeJob>('operations', { kind: 'install', args: downloadValues }); setJobId(result.id); setPlan(null); }); }
    return <Screen plainHeader title={tr("Model")} titleDetail={selectedFamily ? <Select ariaLabel={tr("Model family")} value={selectedFamily.family} menuWidth={340} options={families.map(f => ({ value: f.family, label: f.title }))} onChange={id => { setFamily(id); setSize(''); setVision('no'); setError(''); }}/> : undefined} actions={<Btn icon="sliders" aria-pressed={expert} aria-expanded={expert} aria-controls="native-expert-settings" onClick={() => { setExpert(v => !v); setMultiple(false); setGpus([]); setDeviceChoice('auto'); }}>{tr(expert ? 'Close expert mode' : 'Expert mode')}</Btn>}>
  {tr(selectedFamily ? <div className="native-model-toolbar"><div className="native-toolbar-summary"><Icon name="models"/><strong>{tr(selectedFamily.title)}</strong><small>{tr(selectedSize ? catalogVersionLabel(selectedSize) : '')} · {tr(context / 1024)}{tr("K")}{tr(vision !== 'no' ? ' · Reads pictures' : '')}</small>{tr(native?.instances.some(i => i.model === selectedSize?.id && i.status === 'ready') ? <span className="native-ready">{tr("Ready")}</span> : null)}</div><NativePresets values={values} onApply={v => { const f = families.find(f => f.family === v.family || f.sizes.some(s => s.sourceFamily === v.family)); if (!f)
        return; presetPending.current = true; setFamily(f.family); setSize(f.sizes.find(s => s.model === v.model && (s.sourceFamily === v.family || f.family === v.family))?.id ?? ''); setContext(v.context ?? 32768); setVision(v.vision ?? 'no'); setKv(v.kv ?? 'int8'); setLowRam(v.low_ram ?? 'auto'); }}/><Btn icon="models" onClick={() => setManage(true)}>{tr("Manage models")}</Btn></div> : null)}
  <div className="native-model-layout">
   <div className="stack stack--lg">
    {tr(native?.instances.length && expert ? <section className="st-card native-running"><h2 className="h4">{tr(native.instances.length === 1 ? 'Loaded model' : 'Loaded models')}</h2>{tr(native.instances.map(i => <div key={i.id}><span><strong>{tr(i.label)}</strong><small>{tr(i.gpus.length ? i.gpus.map(g => `GPU ${g + 1}`).join(' + ') : 'Automatic GPU')} · {tr(i.status === 'ready' ? i.phase : i.status)}{tr(native.defaultId === i.id ? ' · Chat default' : '')}</small>{tr(i.error ? <small className="inline-error">{tr(plainFailureText(i.error).message)}</small> : null)}</span>{tr(expert && multiple && native.defaultId !== i.id ? <Btn size="sm" variant="ghost" onClick={() => void perform(() => nativeAction(`instances/${i.id}/default`))}>{tr("Use in chat")}</Btn> : null)}<Btn size="sm" disabled={pending || !!active} onClick={() => void perform(() => nativeAction(`instances/${i.id}/unload`))}>{tr("Unload")}</Btn></div>))}</section> : null)}
    <section className="st-card native-choice">{tr(selectedFamily && !selectedSize ? <><h2 className="h3">{tr("This model needs a newer engine.")}</h2><p className="small muted">{tr("Update your engine or import a local GGUF for this model.")}</p><Btn onClick={() => { setSettingsSection('Engine'); go('settings'); }}>{tr("Update engine")}</Btn><Btn variant="ghost" onClick={onImport ?? onOtherModels}>{tr("Use a local GGUF\u2026")}</Btn></> : null)}{tr(!selectedFamily ? <><h2 className="h3">{tr("Set up Strata")}</h2><p className="small muted">{tr("Connect your local installation to choose an optimized model.")}</p><Btn variant="primary" onClick={onConnectEngine ?? (() => { setSettingsSection('Engine'); go('settings'); })}>{tr("Connect engine")}</Btn>{tr(native?.error ? <p className="small inline-error">{tr(native.error)}</p> : null)}<Btn variant="ghost" onClick={onOtherModels}>{tr("Other local models")}</Btn></> : null)}
     {tr(selectedSize ? <><label className="input-field"><span>{tr("Version")}</span><Select ariaLabel={tr("Model version")} value={selectedSize.id} options={[...selectedFamily!.sizes.map(s => ({ value: s.id, label: `${catalogVersionLabel(s)} · ${s.model} · ${s.download_gb.toFixed(1)} GB${s.installed ? ` · ${tr('Downloaded')}` : ''}` })), { value: 'local-import', label: 'Use a local GGUF…' }]} onChange={id => id === 'local-import' ? (onImport ?? onOtherModels)() : setSize(id)}/></label><p className="small muted">{tr(selectedSize.about)}</p><div className="native-choice-facts"><span><Icon name="memory" size={16}/>{tr(selectedSize.ram_needed_gb)}{tr(" GB RAM class")}</span></div>
      {tr(noFit ? <p className="small muted native-spill-note"><Icon name="disk" size={16}/>{tr("May run slower \u00B7 Model data can be read from SSD.")}</p> : null)}
      {tr(selectedSize?.images ? <div className="native-image-reading"><label><input type="checkbox" checked={vision !== 'no'} disabled={!selectedSize?.images || !!active} onChange={e => setVision(e.target.checked ? 'yes' : 'no')}/><span><strong>{tr("Read pictures ")}<small>{tr("Optional")}</small></strong><small>{tr("Understand photos, screenshots and charts. Vision files download automatically. Enable them when loading this model.")}</small></span></label>{tr(expert && vision !== 'no' ? <Select ariaLabel={tr("Image encoder device")} value={vision === 'yes' ? 'yes' : vision} options={[{ value: 'yes', label: 'Automatic' }, { value: 'gpu', label: 'GPU' }, { value: 'cpu', label: 'CPU' }]} onChange={setVision}/> : null)}</div> : null)}
     </> : null)}
    </section>
    {tr(selectedFamily ? <section className="st-card"><div className="section-toolbar"><h2 className="h4">{tr("Conversation context")}</h2><span className="small muted">{tr(context / 1024)}{tr("K tokens")}</span></div><p className="small muted">{tr("How much your model can keep in mind.")}</p><Select ariaLabel={tr("Conversation context")} value={String(context)} options={(native?.catalog?.contexts ?? [8192, 32768, 65536, 131072, 262144]).map(n => ({ value: String(n), label: `${n / 1024}K` }))} onChange={n => setContext(Number(n))}/></section> : null)}
    <section className="st-card native-computer-use"><h2 className="h4">{tr("Computer use")}</h2><div className="native-resource-choice"><button aria-pressed={resourceProfile === 'work'} data-selected={resourceProfile === 'work'} onClick={() => setResourceProfile('work')}><Icon name="computer" size={20}/><span><strong>{tr("Leave 20% for my work")}</strong><small>{tr("Headroom for your other apps")}</small></span>{tr(resourceProfile === 'work' ? <Icon name="check" size={15}/> : null)}</button><button aria-pressed={resourceProfile === 'performance'} data-selected={resourceProfile === 'performance'} onClick={() => setResourceProfile('performance')}><Icon name="bolt" size={20}/><span><strong>{tr("Full performance")}</strong><small>{tr("Give Strata priority")}</small></span>{tr(resourceProfile === 'performance' ? <Icon name="check" size={15}/> : null)}</button></div></section>
    <details className="native-generation"><summary>{tr("Image generation ")}<span className="muted">{tr("Optional")}</span></summary><RunImageModel library={imageLibrary} disabled={pending || !!active} onChange={() => { }}/></details>
    {tr(expert ? <section id="native-expert-settings" ref={expertSection} className="st-card native-execution"><h2 className="h4">{tr("Expert settings")}</h2><label className="native-toggle"><span><strong>{tr("Run multiple models")}</strong><small>{tr("Loading another model keeps the others loaded.")}</small></span><button className="st-toggle" role="switch" aria-label={tr("Run multiple models")} aria-checked={multiple} onClick={() => setMultiple(v => !v)}/></label><h3 className="small strong">{tr("Run on")}</h3><p className="small muted">{tr("Automatic uses the installation\u2019s GPU placement. Choose a GPU, or split this model across several GPUs.")}</p><Select ariaLabel={tr("Run on")} value={deviceChoice} options={[{ value: 'auto', label: 'Automatic' }, ...(native?.hardware?.gpus ?? []).map(g => ({ value: String(g.index), label: `GPU ${g.index + 1} · ${g.name}` })), ...((native?.hardware?.gpus.length ?? 0) > 1 ? [{ value: 'split', label: 'Split across GPUs…' }] : [])]} onChange={value => { setDeviceChoice(value); setGpus(value === 'auto' ? [] : value === 'split' ? (native?.hardware?.gpus ?? []).map(g => g.index) : [Number(value)]); }}/>{tr(deviceChoice === 'split' ? <div className="native-gpu-options">{tr(native?.hardware?.gpus.map(g => <label key={g.index}><input type="checkbox" checked={gpus.includes(g.index)} onChange={e => setGpus(old => e.target.checked ? [...old, g.index] : old.filter(i => i !== g.index))}/><span>{tr("GPU ")}{tr(g.index + 1)} · {tr(g.name)}<small>{tr(g.vram_gb)}{tr(" GB VRAM")}</small></span></label>))}</div> : null)}{tr(gpus.length > 1 ? <label className="input-field"><span>{tr("Layer split")}</span><input className="st-input" value={split} placeholder={tr("auto or 16,32")} onChange={e => setSplit(e.target.value)}/></label> : null)}<div className="native-field-grid"><label className="input-field"><span>{tr("Context memory")}</span><Select ariaLabel={tr("Native KV format")} value={kv} options={[{ value: 'int8', label: 'INT8' }, { value: 'q4_0', label: 'Q4' }, { value: 'k8v4', label: 'K8 / V4' }]} onChange={setKv}/></label><label className="input-field"><span>{tr("Low-RAM mode")}</span><Select ariaLabel={tr("Native low RAM mode")} value={lowRam} options={['auto', 'resident', 'mmap'].map(v => ({ value: v, label: v === 'auto' ? 'Automatic' : v === 'resident' ? 'Keep experts in RAM' : 'Read experts from disk' }))} onChange={setLowRam}/></label></div>{tr(installed ? <NativeSettings model={installed.model}/> : <p className="small muted">{tr("More engine parameters become available after preparing this version.")}</p>)}</section> : null)}
    {tr(active ? <NativeProgress job={active}/> : null)}
    {tr(job?.state === 'error' ? <ErrorNotice message={job.error ?? job.summary} action={tr("Retry")} onRetry={() => void load()}/> : null)}
    {tr(error ? <ErrorNotice message={error}/> : null)}
    <div className="native-model-actions">{tr(onReady && native?.instances.some(i => i.status === 'ready') ? <Btn variant="primary" onClick={onReady}>{tr("Start chatting")}</Btn> : <><Btn icon="gauge" disabled={!native?.instances.some(i => i.status === 'ready') || !!active} onClick={() => setBenchmark(true)}>{tr("Tune model")}</Btn><Btn variant="primary" icon="models" disabled={pending || !native?.available || !!active || !selectedSize || expert && deviceChoice === 'split' && gpus.length < 2} onClick={() => void load()}>{tr(pending ? 'Checking…' : installed?.ready ? 'Load model' : 'Download & load')}</Btn></>)}</div>
   </div>
   <aside className="stack stack--lg"><section className="st-card"><h2 className="h4">{tr("This computer")}</h2><p>{tr(native?.hardware?.ram_gb || hardware?.memory.totalGb || '—')}{tr(" GB RAM")}</p>{tr(native?.hardware?.gpus.length ? native.hardware.gpus.map(g => <p className="small" key={g.index}>{tr(g.name)} · {tr(g.vram_gb)}{tr(" GB")}</p>) : <p className="small muted">{tr("No supported GPU reported by this installation.")}</p>)}<p className="small muted">{tr("Compatibility and sizes come from your installed Strata version.")}</p></section><section className="st-card"><h2 className="h4">{tr("Model files")}</h2><p className="small muted">{tr("Strata prepares all shards, the draft layer and shared files. Completed downloads are reused.")}</p><code className="native-path">{installed?.dataFolder ?? native?.downloadDirectory ?? native?.installed?.data_folder ?? 'Choose an installation'}</code><Btn size="sm" variant="ghost" onClick={() => { setSettingsSection('Storage'); go('settings'); }}>{tr("Storage settings")}</Btn></section><Btn size="sm" variant="ghost" onClick={onOtherModels}>{tr("Other local models\u2026")}</Btn></aside>
  </div>
  <Dialog open={!!plan} title={tr("Download and prepare")} compact onClose={() => !pending && setPlan(null)} actions={<><Btn disabled={pending} onClick={() => setPlan(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || !!plan?.plan?.disk_short} onClick={() => void download()}>{tr("Download & load")}</Btn></>}><p><strong>{tr(plan?.plan?.title ?? selectedFamily?.title)}</strong>{tr(" \u00B7 approximately ")}{tr(plan?.plan?.download_gb ?? selectedSize?.download_gb)}{tr(" GB")}</p>{tr(plan?.plan?.disk_short ? <ErrorNotice message={plan.plan.disk_short}/> : null)}<p className="small muted">{tr("Existing files are checked and reused. Any currently loaded native models will be unloaded before preparing files.")}</p><p className="small muted">{tr("Your computer may feel slower while weights load. You can stop the download and resume later.")}</p>{tr(selectedSize?.images ? <p className="small muted">{tr("Vision encoder files are included. Picture reading loads only when enabled.")}</p> : null)}</Dialog>
  <Dialog open={manage} title={tr("Manage models")} onClose={() => setManage(false)} actions={<Btn onClick={() => setManage(false)}>{tr("Done")}</Btn>}><div className="stack"><p className="strong">{tr(selectedFamily?.title)}</p><div className="st-tablewrap"><table className="st-table"><thead><tr><th>{tr("Version")}</th><th>{tr("Download")}</th><th>{tr("Status")}</th><th></th></tr></thead><tbody>{tr(selectedFamily?.sizes.map(v => <tr key={v.id}><td><strong>{tr(catalogVersionLabel(v))}</strong><small>{tr(v.model)} · {tr(v.about)}</small></td><td>{tr(v.download_gb.toFixed(1))}{tr(" GB")}</td><td>{tr(native?.instances.some(i => i.model === v.id) ? 'Loaded' : v.installed ? 'Downloaded' : 'Available')}</td><td><Btn size="sm" disabled={!!active} onClick={() => { setSize(v.id); setManage(false); }}>{tr(selectedSize?.id === v.id ? 'Selected' : 'Use version')}</Btn></td></tr>))}</tbody></table></div><Btn variant="ghost" icon="folder" onClick={() => { setManage(false); (onImport ?? onOtherModels)(); }}>{tr("Use a local GGUF\u2026")}</Btn><p className="xs muted">{tr("Choose a version, then load it from Models. Shared native files are kept together in Storage.")}</p></div></Dialog>
  <NativeBenchmark open={benchmark} onClose={() => setBenchmark(false)}/>
 </Screen>;
}
/** setup's "5/7: downloading …" as a sentence a person can read. */
function installStepLabel(step?: string): string {
    const n = Number(step?.match(/^(\d+)\//)?.[1]);
    if (!n)
        return step ? 'Getting things ready…' : 'Downloading and preparing model files…';
    return n <= 2 ? 'Checking your computer…' : n <= 4 ? 'Getting things ready…' : n === 5 ? 'Downloading your model…' : n === 6 ? 'Preparing your model for Strata…' : 'Finishing up…';
}
export function NativeProgress({ job }: {
    job: NativeJob;
}) { const download = installProgress(job.result?.install); const detailsText = [job.details, job.result?.install?.engine?.details].filter(Boolean).join('\n'); const eta = useDownloadEta(job.kind === 'install' ? download : undefined, job.id); const [stopping, setStopping] = useState(false), [error, setError] = useState(''); async function stop() { setStopping(true); setError(''); try {
    await nativeAction(`operations/${job.id}/cancel`);
}
catch (e) {
    setError(e instanceof Error ? e.message : String(e));
}
finally {
    setStopping(false);
} } return <section className="st-card native-progress" aria-live="polite"><div className="section-toolbar"><h2 className="h4">{tr(job.kind === 'install' ? 'Preparing your model' : job.kind === 'load' ? 'Loading model' : 'Engine operation')}</h2>{tr(job.kind === 'install' || job.kind === 'load' ? <div className="onboard-progress-actions">{tr(eta ? <span className="small muted">{eta}</span> : null)}<Btn size="sm" disabled={stopping} onClick={() => void stop()}>{tr(stopping ? 'Stopping…' : 'Stop')}</Btn></div> : null)}</div><p className="small">{tr(job.kind === 'install' ? installStepLabel(job.result?.install?.step) : job.summary)}</p>{tr(download ? <><progress max={100} value={download.percent ?? 0}/><p className="xs muted">{tr(download.file === 'model files' ? 'Model files' : download.file)} · {tr(download.percent ?? 0)}%{tr(Number.isFinite(download.done_gb) && Number.isFinite(download.total_gb) ? ` · ${download.done_gb!.toFixed(1)} / ${download.total_gb!.toFixed(1)} GB` : '')}</p></> : null)}<details><summary className="small">{tr("Details")}</summary><p className="small muted">{tr("Completed files are retained. Downloading, packing and loading are separate stages; readiness is confirmed by the engine.")}</p>{tr(detailsText ? <pre className="small muted error-detail-text">{detailsText}</pre> : null)}</details>{tr(error ? <ErrorNotice message={error}/> : null)}</section>; }
