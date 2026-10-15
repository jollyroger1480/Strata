/**
 * The host's native "choose a folder" dialog, one per platform:
 *   macOS    osascript (choose folder)            -- unchanged
 *   Windows  PowerShell FolderBrowserDialog (STA, topmost owner so it never opens behind the app)
 *   Linux    zenity, then kdialog
 * Where no dialog exists the caller gets { unavailable: true } and tells the person to type a path.
 */
import { stat } from 'node:fs/promises';
import { exec } from './util.mjs';

const MAC_SCRIPT = prompt => `POSIX path of (choose folder with prompt "${prompt.replace(/["\\]/g, '')}")`;

export function windowsFolderScript(prompt) {
  const text = prompt.replace(/'/g, "''");
  return [
    'Add-Type -AssemblyName System.Windows.Forms',
    '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8',
    '$owner = New-Object System.Windows.Forms.Form',
    '$owner.TopMost = $true; $owner.ShowInTaskbar = $false; $owner.StartPosition = "CenterScreen"; $owner.Size = New-Object System.Drawing.Size(0, 0)',
    '$owner.Show(); $owner.Activate()',
    '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog',
    `$dialog.Description = '${text}'`,
    '$dialog.ShowNewFolderButton = $true',
    'if ($dialog.ShowDialog($owner) -eq "OK") { [Console]::Out.Write($dialog.SelectedPath) }',
    '$owner.Close()',
  ].join('; ');
}

/** powershell.exe takes the script as UTF-16LE base64, so no quoting survives to be mangled. */
export function windowsFolderArgs(prompt) {
  return ['-NoProfile', '-NonInteractive', '-STA', '-EncodedCommand', Buffer.from(windowsFolderScript(prompt), 'utf16le').toString('base64')];
}

/** Strip trailing separators, but leave a drive or filesystem root intact. */
export function cleanChosenPath(value) {
  const text = String(value ?? '').trim();
  if (/^[A-Za-z]:[\\/]*$/.test(text)) return text.slice(0, 2) + '\\';
  return text.length > 1 ? text.replace(/[\\/]+$/, '') || text.slice(0, 1) : text;
}

const missing = error => error?.code === 'ENOENT';

/**
 * Ask for a folder. Resolves { path } | { cancelled: true } | { unavailable: true, reason }.
 * `run` is injectable for tests: (command, args) => Promise<{ stdout }>.
 */
export async function chooseFolder(prompt, { platform = process.platform, run = exec } = {}) {
  if (platform === 'darwin') {
    try {
      const { stdout } = await run('osascript', ['-e', MAC_SCRIPT(prompt)]);
      return stdout.trim() ? { path: cleanChosenPath(stdout) } : { cancelled: true };
    } catch {
      return { cancelled: true }; // dismissing the dialog is an exit code, not a failure
    }
  }
  if (platform === 'win32') {
    const shell = `${process.env.SystemRoot || 'C:\\Windows'}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe`;
    try {
      const { stdout } = await run(shell, windowsFolderArgs(prompt));
      return stdout.trim() ? { path: cleanChosenPath(stdout) } : { cancelled: true };
    } catch (error) {
      if (missing(error)) return { unavailable: true, reason: 'Windows PowerShell was not found.' };
      throw new Error('The folder chooser could not open.');
    }
  }
  const candidates = [
    ['zenity', ['--file-selection', '--directory', `--title=${prompt}`]],
    ['kdialog', ['--getexistingdirectory', '.', '--title', prompt]],
  ];
  for (const [command, args] of candidates) {
    try {
      const { stdout } = await run(command, args);
      return stdout.trim() ? { path: cleanChosenPath(stdout) } : { cancelled: true };
    } catch (error) {
      if (missing(error)) continue;
      if (error?.code === 1) return { cancelled: true }; // both tools exit 1 when the dialog is dismissed
      throw new Error('The folder chooser could not open.');
    }
  }
  return { unavailable: true, reason: 'No folder chooser is installed (install zenity or kdialog).' };
}

export const TYPE_A_PATH = 'No folder chooser is available on this system, so type the folder path instead.';

/** A typed path stands in for the dialog: it must be an absolute directory that exists. */
export async function typedFolder(value, { isAbsolute }) {
  const text = typeof value === 'string' ? value.trim() : '';
  if (!text || text.includes('\0')) return null;
  if (!isAbsolute(text)) throw new Error('Enter the full path of the folder.');
  let info;
  try { info = await stat(text); } catch { throw new Error('That folder does not exist.'); }
  if (!info.isDirectory()) throw new Error('Choose a folder, not a file.');
  return cleanChosenPath(text);
}
