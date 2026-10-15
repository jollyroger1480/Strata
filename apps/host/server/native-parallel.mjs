/**
 * Several requests at once (the engine's batch slots) for the installed app.
 *
 * The app is for people who run a chat, a helper agent and a second chat side by side, so a model loads with
 * 4 request slots by default instead of making them queue. Two engine facts decide how far that can go:
 *
 *   - Every slot keeps its OWN conversation state in graphics memory at the model's full context
 *     (docs/BATCHING.md, "What a slot costs"): the context is per slot, so 4 slots cost 4 times one slot.
 *   - Without `--batch-mtp` the slots decode WITHOUT the draft layer. With it, each slot verifies one draft per window
 *     (docs/BATCHING.md), at the price of a little more memory per slot. A request that runs alone still takes the
 *     engine's solo path, so a single chat is not slower for having slots.
 *
 * Recommend, never force: a number the person sets (Expert mode, or a value already in the model's saved settings)
 * is kept exactly. The default only fills in what nobody chose. If the card has no room for 4 it is fewer, and the
 * person is told in plain words; a load that fails with several slots is tried again with one before it is reported.
 *
 * The engine does the final fitting itself ("it never refuses a count it cannot run": it logs what it could do).
 * This module keeps the request realistic so the expert cache is not squeezed for a count that could not run.
 */

export const DEFAULT_SLOTS = 4;
/** The most the engine's batch window holds without grouped MTP. */
export const MAX_SLOTS = 8;
/**
 * THE one-line policy.
 *   'fit'   the default aims for 4 and takes fewer where the slots would take more than SLOT_SHARE of the card's memory
 *           (measured on a 12 GB RTX 3060, 32K: 4 slots took 3.7 GiB, left 0.35 GiB free and cost a single chat 37%)
 *   'fixed' the default is always DEFAULT_SLOTS, whatever the card (the engine still fits what it can)
 */
export const PARALLEL_POLICY = 'fit';
/** Share of ALL the graphics memory the slots' sessions may take together under the 'fit' policy. */
export const SLOT_SHARE = 0.2;

const argValue = (args, flag) => { const at = args.indexOf(flag); return at >= 0 ? args[at + 1] : undefined; };

/** Graphics memory (GB) one slot's session takes at `context` tokens. Flash-Next: 12 attention layers; Qwen3.6: 45 KB per token. */
export function slotGb({ context, kv = 'int8', family = 'flash' }) {
  const ctx = Number(context) > 0 ? Number(context) : 32768;
  if (family === 'q36') return (ctx * 45_000) / 1e9 + 0.07;
  return (ctx * 12 * (kv === 'q4_0' ? 1150 : 2100)) / 1e9 + 0.17; // measured on the 0.1.45 engine: 0.92 GiB per slot at 32K
}

/** What the saved settings say about the engine, read from the config (never from the screen). */
export function engineFacts(config = {}) {
  const args = Array.isArray(config.args) ? config.args.map(String) : [];
  const context = Number(argValue(args, '--max-context') ?? argValue(args, '--context') ?? config.max_context ?? config.context) || null;
  const kv = /q4/i.test(argValue(args, '--kv') ?? argValue(args, '--kv-type') ?? '') ? 'q4_0' : 'int8';
  const family = /qwen3\.6|qwen36|35b-a3b/i.test(`${config.model_name ?? ''} ${args.join(' ')}`) ? 'q36' : 'flash';
  const gpus = Array.isArray(config.gpu) ? config.gpu : config.gpu === undefined || config.gpu === null ? [0] : [config.gpu];
  return {
    args, context, kv, family, gpus,
    mtp: args.includes('--mtp'),
    batchMtp: args.includes('--batch-mtp'),
    explicitBatch: args.includes('--batch') || args.includes('--slots'),
  };
}

/**
 * @param {object} input
 * @param {object} input.config            the model's saved settings (args, gpu, parallel, ...)
 * @param {object} [input.values]          what the person set for this load (Expert mode's "Concurrent requests")
 * @param {object} [input.hardware]        the host's hardware read (`gpus[{index, vram_gb}]`)
 * @param {number} [input.requested]       the default to aim for
 * @returns {{slots:number, batchMtp:boolean, explicit:boolean, requested:number, note:string|null, slotGb:number}}
 */
export function chooseParallel({ config = {}, values = {}, hardware = null, requested = DEFAULT_SLOTS }) {
  const facts = engineFacts(config);
  const slot = slotGb({ context: facts.context, kv: facts.kv, family: facts.family });
  const chosen = values.parallel !== undefined && values.parallel !== null ? Number(values.parallel)
    : config.parallel !== undefined && config.parallel !== null ? Number(config.parallel) : null;
  // A number someone chose (including 1) is the person's decision, and so is an explicit --batch in the engine arguments.
  if (facts.explicitBatch || (chosen !== null && Number.isFinite(chosen))) {
    const slots = facts.explicitBatch ? null : Math.max(1, Math.min(1024, Math.trunc(chosen)));
    return { slots: slots ?? 0, batchMtp: false, explicit: true, requested, note: null, slotGb: slot };
  }
  const cards = (hardware?.gpus ?? []).filter(g => facts.gpus.includes(g.index)).map(g => Number(g.vram_gb)).filter(n => Number.isFinite(n) && n > 0);
  // Without a hardware read the engine's own fitting is all there is: ask for the default and let it say what it can run.
  if (!cards.length) return { slots: requested, batchMtp: facts.mtp && facts.gpus.length === 1, explicit: false, requested, note: null, slotGb: slot };
  const total = cards.reduce((a, b) => a + b, 0);
  let slots = requested;
  if (PARALLEL_POLICY === 'fit') while (slots > 1 && slots * slot > SLOT_SHARE * total) slots -= 1;
  const note = slots < requested
    ? (slots > 1
      ? `Your graphics card has room for ${slots} requests at once with this context size, so Strata set that. A smaller context size leaves room for more.`
      : 'Your graphics card has no spare room for several requests at once with this context size, so Strata set it to one at a time. A smaller context size leaves room for more.')
    : null;
  return { slots, batchMtp: slots > 1 && facts.mtp && !facts.batchMtp && facts.gpus.length === 1, explicit: false, requested, note, slotGb: slot };
}

/**
 * Apply a choice to the run configuration that is about to be saved. Never touches a number the person chose.
 * Returns the configuration and the plain-words note (or null).
 */
export function applyParallel(cfg, choice) {
  if (choice.explicit) return { cfg, note: null };
  const next = { ...cfg };
  if (choice.slots > 1) next.parallel = choice.slots; else delete next.parallel;
  if (choice.batchMtp) next.args = [...(next.args ?? []), '--batch-mtp'];
  return { cfg: next, note: choice.note };
}

/** The same configuration with one request at a time, for the second try after a load that failed with several. */
export function withoutParallel(cfg) {
  const next = { ...cfg };
  delete next.parallel;
  next.args = (next.args ?? []).filter(a => a !== '--batch-mtp');
  return next;
}

/** What to say when the second try worked. */
export const FALLBACK_NOTE = 'The model could not start with several requests at once on this PC, so Strata loaded it for one request at a time. Everything still works; helper agents and a second chat wait their turn.';
