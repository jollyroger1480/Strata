import { tr } from '../lib/i18n';
import { Icon, type IconName } from './ui';
import type { TuneProgress, TuneResult } from '../lib/native-tune';
import { tunePhase, tuneSpeeds } from '../lib/tune-phase';

const STEPS: { icon: IconName; title: string; running: string }[] = [
  { icon: 'cpu', title: 'Checking your PC', running: 'Looking at your graphics card and memory' },
  { icon: 'gauge', title: 'Measuring the base speed', running: 'Timing your current settings' },
  { icon: 'sliders', title: 'Trying settings', running: 'Trying other settings, one at a time' },
  { icon: 'thinking', title: 'Confirming the best one', running: 'Repeating the test to be sure' },
  { icon: 'disk', title: 'Saving', running: 'Keeping it for this model; you can undo it' },
];

const fmt = (n: number) => (Math.round(n * 10) / 10).toFixed(1);

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="tune-spark tune-spark--empty" aria-hidden="true"/>;
  const lo = Math.min(...values), hi = Math.max(...values), span = Math.max(0.0001, hi - lo);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * 220, 56 - ((v - lo) / span) * 46] as const);
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const [lx, ly] = pts[pts.length - 1];
  return <svg className="tune-spark" viewBox="0 0 220 64" preserveAspectRatio="none" aria-hidden="true"><path d={`${line} L220 64 L0 64Z`} className="tune-spark__fill"/><path d={line} className="tune-spark__line"/><circle cx={lx} cy={ly} r="4"/></svg>;
}

/** The tuner at work: five steps, each with its own icon and state, and the speed it has measured so far. */
export function TuneRunView({ progress, ratio, saving }: { progress: TuneProgress | undefined; ratio: number; saving: boolean }) {
  const at = saving ? 4 : tunePhase(progress?.stage);
  const speeds = tuneSpeeds(progress);
  const base = speeds[0]?.tok_s ?? null;
  const now = speeds.length ? speeds[speeds.length - 1].tok_s : null;
  const left = progress?.minutesLeft ?? null;
  const sub = (i: number): string => {
    if (i < at) return i === 1 && base != null ? tr("Starting point: {0} tokens per second", [fmt(base)]) : '';
    if (i === at && progress?.step) return progress.step;
    return i === at ? tr(STEPS[i].running) : '';
  };
  return <div className="tune-run">
    <ol className="tune-steps2" aria-label={tr("Tuning steps")}>
      {STEPS.map((s, i) => {
        const state = i < at ? 'done' : i === at ? 'run' : 'pending';
        return <li key={s.title} className="tune-step" data-s={state} data-step={s.icon} aria-current={state === 'run' ? 'step' : undefined}>
          <span className="tune-step__badge"><Icon name={state === 'done' ? 'check' : s.icon} size={22}/></span>
          <span className="tune-step__text"><strong>{tr(s.title)}</strong><small>{sub(i)}</small></span>
          <span className="tune-step__state">{tr(state === 'done' ? 'Done' : state === 'run' ? 'Running' : 'Waiting')}</span>
        </li>;
      })}
    </ol>
    <aside className="tune-live" aria-live="polite">
      <span className="label">{tr("Speed now")}</span>
      <div className="tune-live__big">{now != null ? fmt(now) : '—'}<small>{tr("tokens per second")}</small></div>
      <Spark values={speeds.map(x => x.tok_s)}/>
      <div className="tune-live__bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(ratio * 100)}><i style={{ width: `${Math.max(3, Math.round(ratio * 100))}%` }}/></div>
      <p className="small muted">{left != null ? tr("about {0} min left", [left]) : tr("This takes about 20 minutes.")}</p>
      <ul className="tune-live__list">{speeds.slice(-6).map((x, i) => <li key={`${x.label}-${i}`}><span>{x.label === 'base' ? tr("Your settings") : x.label}</span><b data-up={base != null && x.label !== 'base' && x.tok_s > base ? 'true' : undefined}>{fmt(x.tok_s)}</b></li>)}</ul>
    </aside>
  </div>;
}

/** The result: the gain large, before and after, and what it was saved as. */
export function TuneResultView({ result, saved }: { result: TuneResult; saved: string | null }) {
  const before = result.base?.decode_tok_s ?? 0, after = result.chosen?.decode_tok_s ?? 0;
  const gain = result.gain_pct ?? (before ? ((after / before) - 1) * 100 : 0);
  const top = Math.max(before, after, 1);
  return <div className="tune-result">
    <div className="tune-hero"><span className="label">{tr("Faster answers")}</span><div className="tune-hero__big">+{Math.round(gain)}%</div>
      <p className="small muted">{tr("{0} to {1} tokens per second", [fmt(before), fmt(after)])}</p></div>
    <div className="tune-compare">
      <div><span className="small muted">{tr("Before")}</span><div className="tune-compare__bar"><i style={{ width: `${(before / top) * 100}%` }}/></div><b>{fmt(before)}</b></div>
      <div data-new="true"><span className="small">{tr("After")}</span><div className="tune-compare__bar"><i style={{ width: `${(after / top) * 100}%` }}/></div><b>{fmt(after)}</b></div>
      {saved ? <p className="small muted">{tr("Saved as {0}. It is used every time you load this model.", [saved])}</p> : null}
    </div>
  </div>;
}
