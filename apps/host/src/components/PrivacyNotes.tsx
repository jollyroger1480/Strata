import { tr } from '../lib/i18n';
import { Icon, type IconName } from './ui';

const NOTES: { icon: IconName; text: string; strong?: boolean }[] = [
  { icon: 'shield', text: 'Strata sends no usage telemetry.', strong: true },
  { icon: 'file', text: 'Conversations stay in browser storage.' },
  { icon: 'link', text: 'When you choose an API model, that conversation and enabled skill instructions are sent to its provider.' },
  { icon: 'activity', text: 'The UI request log keeps client, timing and token metadata in memory.' },
  { icon: 'memory', text: 'Native API monitoring can retain request content in memory when enabled.' },
  { icon: 'tool', text: 'Connected tools access the resources you configure.' },
];

/** What Strata collects and where things go: short lines across the box, nothing long to read through. */
export function PrivacyNotes() {
  return <ul className="privacy-notes" aria-label={tr("Collection and logs")}>
    {NOTES.map(n => <li key={n.text} data-strong={n.strong || undefined}><Icon name={n.icon} size={15}/><span>{tr(n.text)}</span></li>)}
  </ul>;
}
