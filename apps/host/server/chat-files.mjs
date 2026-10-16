/**
 * The folder a chat works in, and what can be done with a file in it.
 *
 *   - Each chat can have its own project folder ("Select a folder" above the input). It becomes the agent's working folder and
 *     sandbox for that chat and is remembered per chat.
 *   - A file in that folder is opened by its PATH (file:///...), never through a local web address: in the browser, in its own app,
 *     shown in the folder (Explorer /select), or copied as a path. Only files inside the chat's folder (or one the agent made
 *     there) can be named, so a request cannot reach anywhere else.
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync, realpathSync } from 'node:fs';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

/** A folder no chat may use as its project: the top of a drive, or the system's own folders. */
export function refusedFolder(dir) {
  const p = resolve(dir);
  if (dirname(p) === p) return 'Choose a folder inside a drive, not the whole drive.';
  if (process.platform === 'win32') {
    const win = (process.env.SystemRoot ?? 'C:\\Windows').toLowerCase();
    const q = p.toLowerCase();
    if (q === win || q.startsWith(win + sep) || /^[a-z]:\\(program files( \(x86\))?|programdata)(\\|$)/.test(q)) return 'That is a system folder. Choose one of your own folders.';
  } else if (/^\/(bin|boot|dev|etc|lib|lib64|proc|root|run|sbin|sys|usr|var)(\/|$)/.test(p)) return 'That is a system folder. Choose one of your own folders.';
  return null;
}

export const isInside = (root, target) => {
  const rel = relative(resolve(root), resolve(target));
  return rel === '' || (!!rel && !rel.startsWith('..') && !isAbsolute(rel));
};

/** The chat -> folder table, kept in one small file. */
export function createChatFolders({ file }) {
  let table = {};
  try { const parsed = JSON.parse(readFileSync(file, 'utf8')); if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) table = parsed; } catch { /* first run */ }
  const save = () => {
    try { mkdirSync(dirname(file), { recursive: true }); const tmp = file + '.tmp'; writeFileSync(tmp, JSON.stringify(table, null, 2)); renameSync(tmp, file); } catch { /* kept in memory */ }
  };
  const alive = p => { try { return statSync(p).isDirectory(); } catch { return false; } };
  return {
    /** The folder of a chat, or null (none chosen, or it was deleted: then the chat falls back to the default). */
    get(chatId) { const p = table[chatId]; return typeof p === 'string' && alive(p) ? p : null; },
    info(chatId) { const p = table[chatId]; return p ? { path: p, name: basename(p) || p, exists: alive(p) } : null; },
    set(chatId, dir) {
      const target = resolve(String(dir ?? '').trim());
      if (!String(dir ?? '').trim() || !isAbsolute(target)) throw Object.assign(new Error('Choose a folder first.'), { code: 'folder-missing' });
      let st; try { st = statSync(target); } catch { throw Object.assign(new Error("That folder doesn't exist."), { code: 'folder-missing' }); }
      if (!st.isDirectory()) throw Object.assign(new Error("That's a file, not a folder."), { code: 'folder-missing' });
      const why = refusedFolder(target);
      if (why) throw Object.assign(new Error(why), { code: 'folder-refused' });
      table[chatId] = target; save();
      return { path: target, name: basename(target) || target, exists: true };
    },
    clear(chatId) { delete table[chatId]; save(); },
  };
}

/** The real path of a file the chat may touch, or throws a plain sentence. Symbolic links are followed before the check. */
export function resolveChatFile(root, rawPath) {
  if (!root) throw Object.assign(new Error('This chat has no folder yet. Choose one first.'), { code: 'no-folder' });
  const given = String(rawPath ?? '').trim();
  if (!given) throw Object.assign(new Error('Which file?'), { code: 'file-missing' });
  const full = isAbsolute(given) ? resolve(given) : resolve(root, given);
  let real; try { real = realpathSync(full); } catch { throw Object.assign(new Error('That file is no longer there.'), { code: 'file-missing' }); }
  let realRoot; try { realRoot = realpathSync(root); } catch { realRoot = resolve(root); }
  if (!isInside(realRoot, real)) throw Object.assign(new Error("That file is outside this chat's folder, so it was not opened."), { code: 'outside' });
  return real;
}

/**
 * Start a program and return at once: detached, no shared output, never waited for.
 * `STRATA_OPENER=record` (tests) records the command instead of starting anything.
 */
export function launch(command, args, { recorder = null } = {}) {
  if (process.env.STRATA_OPENER === 'record' || recorder) { (recorder ?? launch.recorded).push([command, ...args]); return; }
  const child = spawn(command, args, { detached: true, stdio: 'ignore', windowsHide: false });
  child.on('error', () => {});
  child.unref();
}
launch.recorded = [];

/** What "open" means on each system, for the browser, the file's own app, or the folder. */
export function openCommand(how, file, platform = process.platform) {
  const url = pathToFileURL(file).href;
  if (how === 'folder') {
    if (platform === 'win32') return ['explorer.exe', [`/select,${file}`]];
    if (platform === 'darwin') return ['open', ['-R', file]];
    return ['xdg-open', [dirname(file)]];
  }
  if (how === 'browser') {
    if (platform === 'win32') return ['rundll32.exe', ['url.dll,FileProtocolHandler', url]];   // the default browser, handed a file:// address
    if (platform === 'darwin') return ['open', [url]];
    return ['xdg-open', [url]];
  }
  if (platform === 'win32') return ['rundll32.exe', ['url.dll,FileProtocolHandler', file]];
  if (platform === 'darwin') return ['open', [file]];
  return ['xdg-open', [file]];
}

export function openFile(root, rawPath, how = 'app', { platform = process.platform, recorder = null } = {}) {
  if (!['browser', 'app', 'folder'].includes(how)) throw Object.assign(new Error('That way of opening is not known.'), { code: 'bad-how' });
  const file = resolveChatFile(root, rawPath);
  const [cmd, args] = openCommand(how, file, platform);
  launch(cmd, args, { recorder });
  return { ok: true, started: true, path: file, url: pathToFileURL(file).href, how };
}
