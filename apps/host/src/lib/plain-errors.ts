/**
 * Plain words for failures. The host and Control speak in codes and log lines (`server_exited`, exit codes, "component-missing",
 * a Python message); a person needs what happened and what to do next. Every message the UI shows through ErrorNotice or a toast
 * passes through here: a known failure becomes a sentence and a next step, and the original text stays one click away under
 * "Details" for experts. Strings are fixed so every language has them.
 */
import { tr } from './i18n';

export interface PlainFailure { message: string; next?: string; detail?: string; kind: string }

interface Rule { kind: string; match: RegExp; message: string; next?: string; detail?: (m: RegExpMatchArray, raw: string) => string | undefined }

const MEMORY = 'The model stopped while starting, most likely because the PC ran low on memory.';
const MEMORY_NEXT = 'Close other programs and try again, or choose “Leave 20% for my work”. In Expert mode you can also turn on the low-RAM option, so part of the model is read from the SSD.';

/** Order matters: the first rule that matches wins. */
const RULES: Rule[] = [
  { kind: 'picture-reading', match: /vision files|picture reading/i, message: 'Picture reading isn’t available for this model.', next: 'Turn off picture reading and try again.' },
  { kind: 'memory', match: /server_exited|engine stopped|exited before|died before|out of memory|memory guard|\boom\b|killed/i, message: MEMORY, next: MEMORY_NEXT },
  { kind: 'slow-load', match: /loading took more than/i, message: 'Loading is taking too long.', next: 'The PC may be short on memory. Close other programs and try again.' },
  { kind: 'engine-missing', match: /no engine for this pc|engine component|never downloads an engine|engine .*not installed|no strata engine/i, message: 'Strata’s engine for this computer isn’t installed yet.', next: 'It is added when you download a model. Download your model again to add it.' },
  { kind: 'network', match: /cannot reach|could not reach|can't reach|timed out|econn|enotfound|getaddrinfo|urlopen|connection (?:reset|refused|aborted)|network is unreachable/i, message: 'The download couldn’t reach the internet.', next: 'Check your connection, then press Resume download.' },
  { kind: 'integrity', match: /sha-?256|checksum|does not match|did not match|hash mismatch|safety check/i, message: 'A downloaded file didn’t pass Strata’s safety check.', next: 'Press Resume download and Strata will fetch it again.' },
  { kind: 'assistant-missing', match: /agent component|deepseek harness|component-missing|harness .*not (?:installed|found)/i, message: 'The coding assistant isn’t installed on this computer yet.', next: 'Chat works without it.' },
  { kind: 'control', match: /strata control (?:could not start|did not answer|stopped)|control did not answer/i, message: 'Strata couldn’t read its own components.', next: 'Close and reopen Strata. If this keeps happening, reinstall the app.' },
  { kind: 'technical', match: /traceback|\bexit(?:ed)?(?: code)? \d+|errno|ERR_[A-Z_]+|\bat \S+:\d+|\[X\]|\bENOENT\b|\bEPERM\b|\bEACCES\b/i, message: 'Something went wrong.', next: 'Try again. If it keeps happening, open Details and share them when you ask for help.' },
];

/** The message as a person should read it, its next step, and the original text for the Details disclosure. */
export function plainFailure(raw: string | null | undefined): PlainFailure {
  const text = String(raw ?? '').replace(/^\s*Fixture:\s*/i, '').trim();
  for (const rule of RULES) {
    const m = text.match(rule.match);
    if (!m) continue;
    return { kind: rule.kind, message: rule.message, next: rule.next, detail: rule.detail ? rule.detail(m, text) : text };
  }
  return { kind: 'plain', message: text.replace(/^./, c => c.toUpperCase()) };
}

/** Translated copy of a failure, ready to show. */
export function plainFailureText(raw: string | null | undefined) {
  const f = plainFailure(raw);
  return { ...f, message: tr(f.message), next: f.next ? tr(f.next) : undefined };
}
