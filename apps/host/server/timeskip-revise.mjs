#!/usr/bin/env node
/**
 * One revision attempt for Time-skip, performed by the app's own assistant.
 *
 * Strata Control runs this as its `--revise-command`, with the candidate folder (an isolated copy of
 * the user's work) as the working directory, the request in STRATA_TIMESKIP_PROMPT and the attempt
 * number in STRATA_TIMESKIP_ATTEMPT. It asks the host (which owns the model and the Harness) for one
 * agent turn inside that folder and returns when the turn ends. Exit 0 means "an attempt was made",
 * not "it helped": whether it helped is decided by the checks and, failing those, by the person.
 */
const base = process.env.STRATA_HOST_URL;
const secret = process.env.STRATA_INTERNAL_SECRET;
const prompt = process.env.STRATA_TIMESKIP_PROMPT ?? '';
const attempt = Number(process.env.STRATA_TIMESKIP_ATTEMPT ?? 1);

if (!base || !secret || !prompt) {
  console.error('The revision step needs the Strata app to be running.');
  process.exit(2);
}

try {
  const res = await fetch(`${base}/api/internal/timeskip-revise`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-strata-internal': secret },
    body: JSON.stringify({ cwd: process.cwd(), prompt, attempt }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    console.error(body.error ?? `The assistant could not make the change (HTTP ${res.status}).`);
    process.exit(1);
  }
  console.log(String(body.text ?? '').trim());
} catch (e) {
  console.error(`The assistant could not be reached: ${e.message}`);
  process.exit(1);
}
