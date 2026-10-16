import { tr } from '../lib/i18n';
import type { CatalogModel } from '../lib/native-catalog';

/** The model families of the catalog as a switch in the navbar (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B). A single family is just its name. */
export function FamilySwitch({ families, selected, onPick }: { families: CatalogModel[]; selected?: string; onPick: (family: CatalogModel) => void }) {
  const shown = families.filter(f => f.sizes.length);
  if (shown.length < 2) return shown[0] ? <span className="family-switch family-switch--single">{tr(shown[0].title)}</span> : null;
  return <div className="family-switch" role="tablist" aria-label={tr("Model family")}>
    {shown.map(f => <button key={f.family} type="button" role="tab" aria-selected={f.family === selected} tabIndex={f.family === selected ? 0 : -1} onClick={() => onPick(f)}
      onKeyDown={e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); const at = shown.findIndex(x => x.family === selected); onPick(shown[(at + (e.key === 'ArrowRight' ? 1 : shown.length - 1)) % shown.length]); } }}>{tr(f.title)}</button>)}
  </div>;
}
