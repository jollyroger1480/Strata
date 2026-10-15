/**
 * One Time-skip run.
 *
 * A run is either a **developer preview** (the scripted timeline, kept for design review) or a
 * **real job** driven by Strata Control through `/api/timeskip`. When `job` is present the panel
 * renders the job's own phases, decisions and check outcomes; the scripted moments still supply the
 * motion and layout, but they no longer decide anything.
 *
 * `job` being absent is what makes a run a preview. There is no flag to forget to clear.
 */
export interface TimeSkipRun {
  id: string;
  prompt: string;
  startedAt: number;
  completedAt?: number;
  /** The selected result. */
  retainedVersion?: TimeSkipVersion;
  /** Present only for a real job. */
  job?: TimeSkipJob;
}

/** What the App Host reports about a real run. Field names mirror the job's durable record. */
export interface TimeSkipJob {
  id: string;
  phase: 'queued' | 'checkpointing' | 'generating' | 'checking' | 'decision' | 'completed'
       | 'cancelled' | 'failed' | 'interrupted' | 'needs_user_review';
  attempts: { index: number; decision: string | null; reason: string;
              checks: { id: string; outcome: string; kind: string }[] }[];
  checkpoints: { id: string; label: string; digest: string;
                 checks: { id: string; outcome: string; kind: string }[] }[];
  selectedCheckpoint: string | null;
  decision: { outcome?: string; reason?: string; improvements?: string[];
              regressions?: string[]; retained?: string } | null;
  needsUserReview: boolean;
  error: string | null;
  finishedAt?: number | null;
}

export type TimeSkipVersion = 'original' | 'updated';

export const TIME_SKIP_DURATION = 28000;
export const timeSkipTabId = (id: string) => `time-skip:${id}`;

/** null means an ordinary chat message; an empty string means a missing prompt. */
export function timeSkipPrompt(text: string): string | null {
  const match = text.trim().match(/^\/time-skip(?:\s+([\s\S]*))?$/i);
  return match ? (match[1] ?? '').trim() : null;
}

const moments = [
  { at: 0, status: 'Saving', title: 'A good starting point.', body: 'Saving the first draft before exploring a different direction.', reason: 'The original stays available.', chosen: 'a', pass: 1 },
  { at: 2600, status: 'Exploring', title: 'Two possible directions.', body: 'Keeping the original while a candidate fills its gaps.', reason: 'Building B from the saved draft.', chosen: 'a', pass: 1 },
  { at: 6000, status: 'Exploring', title: 'Making the next step clearer.', body: 'The candidate is adding detail without losing the original idea.', reason: 'A focused correction, from the same checkpoint.', chosen: 'a', pass: 1 },
  { at: 9500, status: 'Comparing', title: 'Does the change hold up?', body: 'Comparing the candidate with the original against the same checks.', reason: 'Checking completeness and what was already working.', chosen: 'a', pass: 1 },
  { at: 12500, status: 'Time-skipping', title: 'Taking the stronger direction.', body: 'B fills the gaps and keeps the useful parts of A.', reason: 'A → B · missing detail added, no check regressed.', chosen: 'b', pass: 1 },
  { at: 16000, status: 'Refining', title: 'Can we take it further?', body: 'B is saved. Trying one more focused revision in the other stream.', reason: 'A new candidate starts from B’s checkpoint.', chosen: 'b', pass: 2 },
  { at: 20500, status: 'Comparing', title: 'Shorter isn’t always better.', body: 'The new revision is concise, but loses a useful detail.', reason: 'One previously passing check now fails.', chosen: 'b', pass: 2 },
  { at: 24000, status: 'Keeping B', title: 'Keep the progress we made.', body: 'Discarding the weaker revision and continuing from B.', reason: 'Revision rejected · the stronger checkpoint is kept.', chosen: 'b', pass: 2 },
  { at: TIME_SKIP_DURATION, status: 'Ready', title: 'The stronger version stays.', body: 'Two passes. One useful improvement. The original is still here.', reason: 'B selected · all three example checks pass.', chosen: 'b', pass: 2 },
] as const;

export function timeSkipFrame(run: TimeSkipRun, now: number) {
  const elapsed = run.completedAt ? TIME_SKIP_DURATION : Math.max(0, now - run.startedAt);
  let index = 0;
  for (let i = 1; i < moments.length; i++) if (elapsed >= moments[i].at) index = i;
  return { ...moments[Math.max(0, index)], index, done: elapsed >= TIME_SKIP_DURATION };
}

/** Result selection is only available after the automatic comparison finishes. */
export function withTimeSkipResult(run: TimeSkipRun, version: TimeSkipVersion, now: number): TimeSkipRun {
  return timeSkipFrame(run, now).done ? { ...run, retainedVersion: version } : run;
}

export function timeSkipResultState(run: TimeSkipRun, now: number) {
  if (run.job) return realResultState(run.job);
  const frame = timeSkipFrame(run, now);
  const ready = frame.index >= 3;
  const kept: TimeSkipVersion = frame.done && run.retainedVersion === 'original' ? 'original' : frame.chosen === 'b' ? 'updated' : 'original';
  const status = frame.done ? kept === 'original' ? 'Original restored' : 'Updated result kept'
    : frame.index === 0 ? 'Saving original…'
    : frame.index === 3 || frame.index === 6 ? 'Checking changes…'
    : frame.index >= 7 ? 'Finishing up…' : 'Revising…';
  return { frame, ready, kept, status };
}

/**
 * Result state for a **real** job.
 *
 * The scripted frame index is derived from the job's phase so the existing layout and motion are
 * unchanged, but every value that carries meaning comes from the job:
 *
 *   * `kept` is the job's retained checkpoint, not a scripted 'a'/'b' choice;
 *   * `status` names the actual phase, and says when the evidence was inconclusive rather than
 *     dressing it as a success;
 *   * `unknown` check outcomes are surfaced as unknown. A phase is not a percentage and this
 *     returns none.
 */
function realResultState(job: TimeSkipJob) {
  const done = ['completed', 'cancelled', 'failed', 'interrupted'].includes(job.phase);
  const index =
    job.phase === 'queued' ? 0
    : job.phase === 'checkpointing' ? 0
    : job.phase === 'generating' ? 2
    : job.phase === 'checking' ? 4
    : job.phase === 'decision' ? 5
    : 8;
  // The decision's `improvements` is the job's own answer about what got better; nothing here
  // re-derives a verdict from the attempt list.
  const improved = (job.decision?.improvements?.length ?? 0) > 0;
  const kept: TimeSkipVersion = improved ? 'updated' : 'original';
  const outcome = job.decision?.outcome ?? null;

  let status: string;
  if (job.phase === 'failed') status = 'The revision could not be completed';
  else if (job.phase === 'cancelled') status = 'Stopped · the saved version was kept';
  else if (job.phase === 'interrupted') status = 'Interrupted · the saved version was kept';
  else if (!done) {
    status = job.phase === 'checkpointing' ? 'Saving the current version…'
      : job.phase === 'generating' ? 'Revising…'
      : job.phase === 'checking' ? 'Checking the changes…'
      : job.phase === 'decision' ? 'Deciding from the checks…'
      : 'Starting…';
  } else if (job.needsUserReview || outcome === 'inconclusive' || outcome === 'conflict') {
    status = 'Ready to compare · the checks were inconclusive';
  } else if (outcome === 'accepted') {
    status = 'Update kept · the checks improved';
  } else {
    status = 'Saved version kept';
  }

  const frame = { ...moments[Math.max(0, Math.min(moments.length - 1, index))], index, done,
                  pass: job.attempts.length > 1 ? 2 : 1 };
  return { frame, ready: index >= 3, kept, status };
}

/**
 * Whether a run still needs a person to choose.
 *
 * The panel uses this to offer Compare instead of claiming a result. It is deliberately not the same
 * question as "is the run finished".
 */
export function timeSkipNeedsReview(run: TimeSkipRun): boolean {
  return Boolean(run.job?.needsUserReview);
}

/** The job's own check outcomes for a checkpoint, for the details table. */
export function timeSkipCheckOutcomes(run: TimeSkipRun): { id: string; outcome: string }[] {
  const job = run.job;
  if (!job) return [];
  const checkpoint = job.checkpoints.find(c => c.id === job.selectedCheckpoint) ?? job.checkpoints[0];
  return (checkpoint?.checks ?? []).map(c => ({ id: c.id, outcome: c.outcome }));
}

export interface TimeSkipDocument {
  title: string;
  steps: readonly string[];
  successCheck: string;
  code?: string;
  image?: { url: string; alt: string };
}

/** Immutable paired sample outputs, separate from the later rejected attempt. */
export function timeSkipDocuments(prompt: string): Record<TimeSkipVersion | 'revision', TimeSkipDocument> & { changes: string[] } {
  if (/image|photo|poster|illustration/i.test(prompt)) return {
    original: { title: 'Fieldnotes cover.svg', steps: [], successCheck: 'Not defined', image: { url: '/examples/fieldnotes-cover.svg', alt: 'Original Fieldnotes cover: Room to begin again, with botanical shapes.' } },
    updated: { title: 'Fieldnotes cover.svg', steps: [], successCheck: 'Keep the botanical style and make the headline easier to scan.', image: { url: '/examples/fieldnotes-cover-timeskip.svg', alt: 'Updated Fieldnotes cover: Make room for ideas, with more open space around the headline.' } },
    revision: { title: 'Fieldnotes cover.svg', steps: [], successCheck: 'Keep the botanical style and make the headline easier to scan.', image: { url: '/examples/fieldnotes-cover-revision.svg', alt: 'Rejected Fieldnotes cover revision: the shortened layout removes the useful subtitle.' } },
    changes: ['Shortened the headline', 'Opened up the composition'],
  } satisfies Record<TimeSkipVersion | 'revision', TimeSkipDocument> & { changes: string[] };
  if (/website|app\b|code|voxel|html|dashboard/i.test(prompt)) return {
    original: { title: 'Counter component', steps: [], successCheck: 'Not defined', code: '<section class="counter">\n  <output>0</output>\n  <button onclick="add()">Add one</button>\n  <button>Reset</button>\n</section>\n<style>\n  .counter { width: 600px; }\n</style>\n<script>\n  const value = document.querySelector("output");\n  function add() { value.value = +value.value + 1; }\n</script>' },
    updated: { title: 'Counter component', steps: [], successCheck: 'Add twice, reset to 0, then check at 320 px.', code: '<section class="counter">\n  <output>0</output>\n  <button onclick="add()">Add one</button>\n  <button onclick="reset()">Reset</button>\n</section>\n<style>\n  .counter { width: min(600px, 100%); }\n</style>\n<script>\n  const value = document.querySelector("output");\n  function add() { value.value = +value.value + 1; }\n  function reset() { value.value = 0; }\n</script>' },
    revision: { title: 'Counter component', steps: [], successCheck: 'Add twice, reset to 0, then check at 320 px.', code: '<section class="counter">\n  <output>0</output>\n  <button onclick="add()">Add one</button>\n  <button>Reset</button>\n</section>\n<style>\n  .counter { width: min(600px, 100%); }\n</style>\n<script>\n  const value = document.querySelector("output");\n  function add() { value.value = +value.value + 1; }\n</script>' },
    changes: ['Connected Reset to clear the counter', 'Made the layout fit small screens', 'Added a repeatable success check'],
  } satisfies Record<TimeSkipVersion | 'revision', TimeSkipDocument> & { changes: string[] };
  return {
    original: { title: 'Launch plan', steps: ['Invite a few testers', 'Collect feedback', 'Make improvements'], successCheck: 'Not defined' },
    updated: { title: 'Launch plan', steps: ['Invite 10 testers', 'Collect feedback by Friday', 'Fix the top 3 issues'], successCheck: '8 of 10 testers complete setup.' },
    revision: { title: 'Launch plan', steps: ['Invite 10 testers', 'Fix the top 3 issues'], successCheck: '8 of 10 testers complete setup.' },
    changes: ['Made all 3 steps specific', 'Added a success check'],
  } satisfies Record<TimeSkipVersion | 'revision', TimeSkipDocument> & { changes: string[] };
}

export function timeSkipDocumentMarkdown(document: TimeSkipDocument) {
  return `# ${document.title}\n\n${document.image ? `![${document.image.alt}](${document.image.url})` : document.code ? `\`\`\`html\n${document.code}\n\`\`\`` : document.steps.map((step, i) => `${i + 1}. ${step}`).join('\n')}\n\n## Success check\n\n${document.successCheck}\n\n*Time-skip demo · sample output, simulated checks.*`;
}

export function timeSkipExample(prompt: string) {
  if (/image|photo|poster|illustration/i.test(prompt)) return {
    labels: ['Visual style', 'Headline', 'Composition'],
    original: ['Botanical direction', 'Longer phrase', 'Dense text area'],
    improved: ['Botanical style kept', 'Shorter and clearer', 'More open space'],
    revision: ['Visual style kept', 'Shortened further', 'Useful subtitle lost'],
    correction: 'Shorten the headline and open up the composition while keeping the botanical style.',
    rejection: 'The later revision removes the useful subtitle. Keep the complete cover.',
  };
  if (/website|app\b|code|voxel|html|dashboard/i.test(prompt)) return {
    labels: ['Core layout', 'Controls', 'Small screens'],
    original: ['Structure in place', 'Reset needs work', 'Content overflows'],
    improved: ['Structure preserved', 'Reset works', 'Fits the viewport'],
    revision: ['Layout simplified', 'Reset lost its state', 'Still fits'],
    correction: 'Fix reset and mobile fit while preserving the layout.',
    rejection: 'The shorter revision breaks reset. Keep B’s working controls.',
  };
  return {
    labels: ['Main idea', 'Next steps', 'Success check'],
    original: ['Clear starting point', 'Too general', 'Not defined yet'],
    improved: ['Original idea kept', 'Specific and ordered', 'Clear way to verify'],
    revision: ['More concise', 'Key detail removed', 'Check preserved'],
    correction: 'Make the steps actionable and add a way to check the result.',
    rejection: 'The shorter revision removes a useful step. Keep B’s complete version.',
  };
}
