import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import type { TimeSkipRun } from '../lib/time-skip';
import { activityWords, phaseFraction } from '../lib/time-skip-activity';
import { stopTimeSkip } from '../lib/store';
import { Btn } from './ui';

const clock = (ms: number) => { const t = Math.max(0, Math.round(ms / 1000)); return t < 60 ? `${t}s` : `${Math.floor(t / 60)}m ${String(t % 60).padStart(2, '0')}s`; };

/**
 * While a Time-skip run works, in the chat: the saved version (A) and the one being written (B) with a stream of light between them, what
 * the assistant is doing right now, how long, and Stop. It is shown from the first moment, so a long run never looks like nothing happening.
 */
export function TimeSkipWorking({ run, sessionId }: { run: TimeSkipRun; sessionId: string }) {
  const [now, setNow] = useState(() => Date.now());
  const [stopping, setStopping] = useState(false);
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  const fraction = phaseFraction(run.job?.phase);
  return <section className="tsw" aria-label={tr("Time-skip is working")}>
    <div className="tsw__stage" aria-hidden="true">
      <div className="tsw__ver"><span>A · {tr("Original")}</span><i style={{ width: '92%' }}/><i style={{ width: '100%' }}/><i style={{ width: '78%' }}/><i style={{ width: '96%' }}/></div>
      <div className="tsw__flow"><i/><i/><i/></div>
      <div className="tsw__ver tsw__ver--b"><span>B · {tr("Revising")}</span><i style={{ width: '70%' }}/><i style={{ width: '88%' }}/><i style={{ width: '55%' }}/></div>
    </div>
    <div className="tsw__bar">
      <div className="tsw__step" role="status"><strong>{activityWords(run.job)}</strong><span className="small muted">{clock(now - run.startedAt)}</span>
        <div className="tsw__prog" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}><i style={{ width: `${Math.round(fraction * 100)}%` }}/></div></div>
      <Btn size="sm" icon="stop" disabled={stopping || !run.job?.id} onClick={() => { setStopping(true); void stopTimeSkip(sessionId, run.id).finally(() => setStopping(false)); }}>{tr("Stop")}</Btn>
    </div>
  </section>;
}
