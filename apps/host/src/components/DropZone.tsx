import { useRef, useState, type ReactNode } from 'react';
import { tr } from '../lib/i18n';
import { Icon } from './ui';

/** Files dropped on the chat go where the attach button would put them: the composer reads the 'strata:drop-files' event. */
export const DROP_EVENT = 'strata:drop-files';
/** The pictures on the clipboard, named so each one is told apart in the composer (a pasted screenshot arrives as "image.png"). */
export function pastedFiles(data: DataTransfer | null | undefined): File[] {
  const out: File[] = [];
  for (const file of Array.from(data?.files ?? [])) {
    if (!file.type.startsWith('image/')) continue;
    const extension = file.type.split('/')[1]?.replace('jpeg', 'jpg').replace(/[+].*/, '') || 'png';
    out.push(new File([file], `Pasted picture ${new Date().toLocaleTimeString().replace(/[^0-9]+/g, '-').replace(/-$/, '')}${out.length ? `-${out.length + 1}` : ''}.${extension}`, { type: file.type }));
  }
  return out;
}
const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files');

/** The conversation column, with a calm highlight while something is dragged over it. */
export function DropZone({ children, className }: { children: ReactNode; className?: string }) {
  const [over, setOver] = useState(false);
  const depth = useRef(0);
  return <div className={className} data-dropping={over || undefined}
    onDragEnter={e => { if (!hasFiles(e)) return; e.preventDefault(); depth.current++; setOver(true); }}
    onDragOver={e => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
    onDragLeave={e => { if (!hasFiles(e)) return; depth.current = Math.max(0, depth.current - 1); if (!depth.current) setOver(false); }}
    onDrop={e => { if (!hasFiles(e)) return; e.preventDefault(); depth.current = 0; setOver(false); const files = Array.from(e.dataTransfer.files); if (files.length) window.dispatchEvent(new CustomEvent(DROP_EVENT, { detail: files })); }}>
    {children}
    {over ? <div className="dropzone" aria-hidden="true"><Icon name="attach" size={22}/><strong>{tr("Drop files to add them")}</strong><span>{tr("Pictures and text files")}</span></div> : null}
  </div>;
}
