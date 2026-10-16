import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useLocation, navigate } from '../lib/router';
import { FloatingMenu } from '../components/FloatingMenu';
import { useEffect, useRef, useState } from 'react';
import { Btn, Dialog, Icon, IconBtn, Select, ErrorNotice } from '../components/ui';
import { Markdown } from '../components/Markdown';
import { Screen } from '../components/layout';
import { api } from '../lib/client';
import { useStore } from '../lib/store';
import { blankTask as blank, type Task } from '../lib/tasks';
import { TaskEditor } from '../components/TaskEditor';
import { TaskDetail, taskState } from '../components/TaskDetail';
import { useNative } from '../lib/native';
const templates = [
    { title: 'Daily briefing', icon: 'sun' as const, prompt: 'Write a short briefing from the notes below. Highlight decisions, open questions and next actions.\n\n[Add your notes here]', frequency: 'weekdays', at: '08:00', detail: 'Turn your notes into a focused morning plan.' },
    { title: 'Weekly review', icon: 'check' as const, prompt: 'Review the updates below. Summarize progress, risks and priorities for next week.\n\n[Add your updates here]', frequency: 'weekly', at: '16:00', detail: 'Reflect on progress and plan the next week.' },
    { title: 'Writing practice', icon: 'pencil' as const, prompt: 'Give me one short writing exercise and an example answer. Keep it practical and under 200 words.', frequency: 'daily', at: '09:00', detail: 'A small daily exercise to sharpen your writing.' },
    { title: 'Learning prompt', icon: 'thinking' as const, prompt: 'Teach me one useful concept about statistics. Include a simple example and one question to check my understanding.', frequency: 'daily', at: '18:00', detail: 'Learn one useful concept at a time.' },
];
export function TasksScreen() {
    const location = useLocation();
    const library = useStore(s => s.library);
    const loaded = library?.variants.find(v => v.id === library.loaded);
    const [tasks, setTasks] = useState<Task[]>([]), [query, setQuery] = useState(''), [sort, setSort] = useState('next'), [archived, setArchived] = useState(false);
    const [editor, setEditor] = useState<Task | null>(null), [assistant, setAssistant] = useState(false), [menu, setMenu] = useState(false), [pending, setPending] = useState(false), [error, setError] = useState(''), [ready, setReady] = useState(false), [result, setResult] = useState<string | null>(null), [removeTask, setRemoveTask] = useState(false);
    const trigger = useRef<HTMLButtonElement>(null);
    async function refresh() {
        const data = await api.get<{
            tasks: Task[];
        }>('/api/tasks');
        setTasks(data.tasks);
        setReady(true);
    }
    useEffect(() => {
        let alive = true;
        const load = () => {
            if (alive)
                void refresh().catch(e => {
                    if (alive) {
                        setError(e.message);
                        setReady(true);
                    }
                });
        };
        load();
        const timer = setInterval(load, 3000);
        window.addEventListener('strata:tasks', load);
        return () => { alive = false; clearInterval(timer); window.removeEventListener('strata:tasks', load); };
    }, []);
    async function action(fn: () => Promise<unknown>) {
        setPending(true);
        setError('');
        try {
            await fn();
            await refresh();
            window.dispatchEvent(new Event('strata:tasks'));
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    function open(task: Task, assist = false) { setEditor({ ...task }); setAssistant(assist); setMenu(false); setError(''); }
    useEffect(() => {
        if (location.query.has('new'))
            open(blank());
    }, [location.query.get('new')]);
    function closeEditor() {
        setEditor(null);
        if (location.query.has('new'))
            navigate('tasks', location.param ?? undefined);
    }
    const currentTask = tasks.find(t => t.id === location.param);
    const visible = tasks.filter(t => (!location.param || t.id === location.param) && !!t.archived === archived && `${t.title} ${t.prompt}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'name' ? a.title.localeCompare(b.title) : (a.nextRunAt ?? 'z').localeCompare(b.nextRunAt ?? 'z'));
    const selected = tasks.find(t => t.id === result);
    const nativeModel = useNative()?.instances.find(i => i.status === 'ready');
    const taskHeader = currentTask ? <div className="task-header-actions">
      <IconBtn name={currentTask.enabled ? 'pause' : 'play'} title={tr(currentTask.enabled ? "Pause {0}" : "Resume {0}", [currentTask.title])} disabled={pending || currentTask.status === 'running' || !!currentTask.archived} onClick={() => void action(() => api.patch(`/api/tasks/${currentTask.id}`, { enabled: !currentTask.enabled }))}/>
      <IconBtn name="trash" title={tr("Delete {0}", [currentTask.title])} disabled={pending || !!currentTask.archived} onClick={() => setRemoveTask(true)}/>
      <Btn icon="pencil" disabled={pending || currentTask.status === 'running'} onClick={() => open(currentTask)}>{tr("Edit")}</Btn>
      <Btn variant="primary" icon={currentTask.status === 'running' ? 'stop' : 'bolt'} disabled={pending || !loaded && currentTask.status !== 'running'} onClick={() => void action(() => api.post(`/api/tasks/${currentTask.id}/${currentTask.status === 'running' ? 'stop' : 'run'}`))}>{tr(currentTask.status === 'running' ? 'Stop' : 'Run now')}</Btn>
    </div> : null;
    return <Screen title={currentTask ? currentTask.title : "Tasks"} back={currentTask ? { label: 'Tasks', onClick: () => navigate('tasks') } : undefined} titleDetail={currentTask ? <span className="task-state" data-state={taskState(currentTask)}>{tr(({ running: 'Running…', active: 'Active', paused: 'Paused' } as const)[taskState(currentTask)])}</span> : undefined} sub={currentTask ? undefined : "Run a prompt on a schedule, or whenever you need it."} actions={currentTask ? taskHeader : <div className="tasks-actions"><input className="st-input tasks-search" type="search" aria-label={tr("Search tasks")} placeholder={tr("Search tasks…")} value={query} onChange={e => setQuery(e.target.value)}/><Select ariaLabel={tr("Sort tasks")} value={sort} options={[{ value: 'next', label: 'Next run' }, { value: 'name', label: 'Name' }]} onChange={setSort}/><button ref={trigger} className="st-btn st-btn--primary" aria-haspopup="menu" aria-expanded={menu} onClick={() => setMenu(v => !v)}>{tr("New task")}<Icon name="chevron" size={16}/></button></div>}>
    <div className="tasks-page">
    
    {tr(!currentTask && <p className="task-runtime small muted"><Icon name="clock" size={16}/>{tr(library?.mode === 'simulated' ? 'Example model · sample answers. ' : '')}{tr("Runs while Strata\u2019s host is open. Runs saved instructions using the loaded model.")}</p>)}
    {tr(!currentTask && <label className="row small muted"><input type="checkbox" checked={archived} onChange={e => setArchived(e.target.checked)}/>{tr("Deleted tasks")}</label>)}
    {tr(error && !editor ? <ErrorNotice message={error} onRetry={() => void refresh()}/> : null)}
    {tr(currentTask ? null : !ready ? <p className="muted">{tr("Loading tasks\u2026")}</p> : !visible.length ? <div className="tasks-empty"><Icon name="clock" size={72}/><h2 className="h3">{tr(query ? 'No matching tasks' : archived ? 'No deleted tasks' : 'No scheduled tasks yet')}</h2><p className="muted">{tr(query ? 'Try another search.' : 'Start with an idea below, or create your own.')}</p></div> : <div className="task-list">{tr(visible.map(t => <article key={t.id} className="task-row"><span className="task-icon"><Icon name="clock" size={24}/></span><div className="task-copy"><h2 className="h4">{t.title}</h2><p className="small muted">{tr(t.status === 'running' ? 'Running…' : t.enabled ? (t.nextRunAt ? `Next: ${new Date(t.nextRunAt).toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' })}` : 'Scheduled · calculating next run') : 'Paused')} · {tr(t.timeZone)}</p>{tr(t.error ? <button className="task-failed text-action" onClick={() => setResult(t.id!)}>{tr("Last run failed \u00B7 View")}</button> : null)}{tr(t.output ? <button className="text-action" onClick={() => setResult(t.id!)}>{tr("View last result")}{tr(t.demo ? ' · Example' : '')}</button> : null)}</div><div className="row">{tr(t.archived ? <Btn size="sm" onClick={() => void action(() => api.patch(`/api/tasks/${t.id}`, { archived: false, enabled: false }))}>{tr("Restore")}</Btn> : <><IconBtn name="pencil" title={tr("Edit {0}", [t.title])} disabled={pending || t.status === 'running'} onClick={() => open(t)}/><Btn size="sm" icon={t.status === 'running' ? 'stop' : 'bolt'} title={tr(t.status === 'running' ? `Stop ${t.title}` : `Run ${t.title} now`)} disabled={pending || !loaded} onClick={() => void action(() => api.post(`/api/tasks/${t.id}/${t.status === 'running' ? 'stop' : 'run'}`))}>{tr(t.status === 'running' ? 'Stop' : 'Run now')}</Btn><label className="row small"><span>{tr("On schedule")}</span><button className="st-toggle" role="switch" aria-label={tr("Schedule {0}", [t.title])} aria-checked={t.enabled} disabled={pending || t.status === 'running'} onClick={() => void action(() => api.patch(`/api/tasks/${t.id}`, { enabled: !t.enabled }))}/></label></>)}</div></article>))}</div>)}
    {tr(currentTask ? <TaskDetail task={currentTask} modelName={nativeModel?.label ?? null}/> : null)}
    {tr(!tasks.some(t => !t.archived) && !query && !archived ? <section className="task-templates" aria-label={tr("Task ideas")}>{tr(templates.map(t => <button key={t.title} className="task-template" onClick={() => open({ ...blank(), ...t, title: tr(t.title), prompt: tr(t.prompt), weekday: 5 })}><span className="task-icon"><Icon name={t.icon} size={24}/></span><span><strong>{tr(t.title)}</strong><span className="small muted">{tr(t.detail)}</span><span className="xs muted"><Icon name="clock" size={13}/>{tr(t.frequency === 'weekly' ? 'Fridays' : t.frequency === 'weekdays' ? 'Weekdays' : 'Daily')} · {tr(t.at)}</span></span></button>))}</section> : null)}
    </div>
    {tr(menu ? <FloatingMenu anchor={trigger.current} title={tr("New task")} onClose={() => setMenu(false)}><button role="menuitem" className="menu__item" disabled={!loaded} onClick={() => open(blank(), true)}><Icon name="thinking"/>{tr("Create with AI")}</button><button role="menuitem" className="menu__item" onClick={() => open(blank())}><Icon name="sliders"/>{tr("Set up manually")}</button></FloatingMenu> : null)}
    {tr(editor ? <TaskEditor task={editor} assistant={assistant} onClose={closeEditor}/> : null)}
    <Dialog open={!!result} title={selected ? <span>{selected.title}</span> : tr('Task result')} onClose={() => setResult(null)} sheet>{tr(selected ? <><p className="small muted">{tr(selected.lastRunAt ? new Date(selected.lastRunAt).toLocaleString(localeTag()) : '')}{tr(selected.demo ? ' · Example result' : '')}</p><div className="document-markdown">{tr(selected.error ? <ErrorNotice message={selected.error} next={tr("Try running this task again.")}/> : <Markdown text={selected.output ?? tr('No result yet.')}/>)}</div></> : null)}</Dialog>
    <Dialog open={removeTask && !!currentTask} compact title={tr("Delete this task?")} onClose={() => !pending && setRemoveTask(false)} actions={<><Btn disabled={pending} onClick={() => setRemoveTask(false)}>{tr("Keep task")}</Btn><Btn variant="danger" disabled={pending} onClick={() => { const id = currentTask?.id; void action(() => api.patch(`/api/tasks/${id}`, { archived: true, enabled: false })).then(() => { setRemoveTask(false); navigate('tasks'); }); }}>{tr("Delete task")}</Btn></>}><div className="stack stack--sm"><p><strong>“{currentTask?.title}”</strong>{tr(" will stop running and be removed from your active tasks.")}</p><p className="small muted">{tr("Your instructions and last result are kept in Deleted tasks. You can restore the task there; its schedule will stay paused.")}</p></div></Dialog>
  </Screen>;
}
