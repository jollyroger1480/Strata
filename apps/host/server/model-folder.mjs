/**
 * Where the person's models go: what a picked folder means (0.2.2).
 *
 * - A drive on its own (D:\ or /) is never filled with model files: it becomes D:\Strata (the data folder) and D:\Strata\Models (the
 *   model library), which the person is told.
 * - The data folder is the person's own choice, so it is written into the settings file Strata Control reads (`data_dir`). Control's
 *   installer accepts a data folder that is that setting, whatever the folder is called; for a folder it has not been told about it wants a
 *   name starting with "Strata" (a rule against an AI tool filling an arbitrary folder, tools/strata_mcp.py check_data_dir).
 * - The link that makes `<data>/models` and `<data>/Models` one folder is only needed where the file system tells them apart; where it cannot be
 *   made (exFAT, FAT, a network share) the models simply go in `<data>/models`.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, symlinkSync, writeFileSync } from 'node:fs';
import { dirname, join, parse, resolve } from 'node:path';

export const isDriveRoot = path => { const p = resolve(path); return p === parse(p).root; };

/** The data folder for a picked folder: a drive root gets a `Strata` folder of its own. `{ path, note }`. */
export function dataFolderFromPick(picked) {
  const path = resolve(picked);
  if (!isDriveRoot(path)) return { path, note: null };
  const target = join(path, 'Strata');
  return { path: target, note: `${path} is a whole drive, so the models go in ${target}.` };
}

/** The model library folder for a picked folder: a drive root gets `Strata/Models`. `{ path, note }`. */
export function libraryFolderFromPick(picked) {
  const path = resolve(picked);
  if (!isDriveRoot(path)) return { path, note: null };
  const target = join(path, 'Strata', 'Models');
  return { path: target, note: `${path} is a whole drive, so the models go in ${target}.` };
}

/** Make a folder the person asked for; a refusal of the disk is said in words. */
export function makeFolder(path) {
  try { mkdirSync(path, { recursive: true }); }
  catch (error) { throw new Error(`Strata could not create ${path} (${error.code === 'EACCES' || error.code === 'EPERM' ? 'access was denied' : error.code === 'ENOSPC' ? 'the drive is full' : error.message}). Choose another folder.`); }
  return path;
}

/** `models` beside `Models`: a link where the two are different names. A disk that cannot hold the link gets a plain `models` folder instead. */
export function alignModels(data, library, { link = symlinkSync } = {}) {
  const models = join(data, 'models');
  if (existsSync(models)) return { data, linked: false };
  try { link(process.platform === 'win32' ? library : 'Models', models, process.platform === 'win32' ? 'junction' : 'dir'); return { data, linked: true }; }
  catch { /* exFAT, FAT, NFS, a share: no links there */ }
  try { mkdirSync(models, { recursive: true }); } catch { /* the install says what it cannot write */ }
  return { data, linked: false };
}

/** Write the person's data folder into the Control settings file (merged, atomically); returns whether it changed. Never throws: the folder
 *  still works for the app, only the Terminal and the installer's own check lose the shared setting. */
export function rememberDataFolder(file, folder) {
  if (!file || !folder) return false;
  try {
    let current = {};
    try { const parsed = JSON.parse(readFileSync(file, 'utf8').replace(/^\uFEFF/, '')); if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) current = parsed; } catch { /* a new file */ }
    if (current.data_dir === folder) return false;
    mkdirSync(dirname(file), { recursive: true });
    const temp = `${file}.${process.pid}.tmp`;
    writeFileSync(temp, JSON.stringify({ ...current, data_dir: folder }, null, 1) + '\n', { mode: 0o600 });
    renameSync(temp, file);
    return true;
  } catch { return false; }
}
