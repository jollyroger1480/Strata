import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n';

/** Minutes left from recent samples of "GB done": the rate over the last minute and a half, so a stall or a burst does not
 *  swing the estimate. Null until there is enough to say (about 10 s of samples) or when nothing is moving. */
export function etaMinutes(samples: { at: number; done: number }[], remainingGb: number | undefined): number | null {
  if (!Number.isFinite(remainingGb) || samples.length < 2) return null;
  const first = samples[0], last = samples[samples.length - 1];
  const seconds = (last.at - first.at) / 1000, gained = last.done - first.done;
  if (seconds < 10 || gained <= 0) return null;
  return Math.max(1, Math.ceil((remainingGb! / (gained / seconds)) / 60));
}

export function etaLabel(minutes: number | null): string {
  if (minutes === null) return '';
  if (minutes >= 90) { const h = Math.floor(minutes / 60), m = minutes % 60; return `${tr('About')} ${h} ${tr('h')}${m ? ` ${m} ${tr('min')}` : ''} ${tr('left')}`; }
  return `${tr('About')} ${minutes} ${tr('min left')}`;
}

/** The approximate time left of a running download, for the progress bar. `key` restarts the estimate for another job. */
export function useDownloadEta(download: { done_gb?: number; remaining_gb?: number } | undefined, key: string | undefined): string {
  const samples = useRef<{ at: number; done: number }[]>([]);
  const owner = useRef<string | undefined>(undefined);
  const [label, setLabel] = useState('');
  const done = download?.done_gb, remaining = download?.remaining_gb;
  useEffect(() => {
    if (owner.current !== key) { owner.current = key; samples.current = []; }
    if (!Number.isFinite(done)) { setLabel(''); return; }
    const now = Date.now();
    const list = samples.current;
    if (!list.length || list[list.length - 1].done !== done) list.push({ at: now, done: done! });
    while (list.length > 2 && now - list[0].at > 90_000) list.shift();
    setLabel(etaLabel(etaMinutes(list, remaining)));
  }, [done, remaining, key]);
  return label;
}
