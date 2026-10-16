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
 *   'fit'   the default aims for 4 and takes fewer where the slots would eat more than CACHE_LOSS_SHARE of the expert cache
 *           the card would otherwise have (the owner's choice: "fit to the card")
 *   'fixed' the default is always DEFAULT_SLOTS, whatever the card (the engine still fits what it can)
 *
 * Why a cache budget: on a 12 GB RTX 3060 (IQ3_XXS, 32K) the slots' sessions come out of the expert cache. Measured, one request
 * alone: no slots 61 tok/s (cache 5.1 GiB); 2 slots 41 (cache 3.25 GiB); 4 slots 38 (cache 1.4 GiB). Several requests together
 * were SLOWER in total than queueing them (52-57 tok/s total queued, against 33-42 in slots). Slots start everyone at once;
 * they pay off when the cache stays mostly intact.
 */
export const PARALLEL_POLICY = 'fit';
/** The share of the solo expert cache the slots' sessions may take together under the 'fit' policy. */
export const CACHE_LOSS_SHARE = 0.2;
/**
 * The expert cache a card has with one request at a time is what remains of its memory after the dense weights, the draft layer,
 * one session, the safety reserve and the prompt path (measured: a 12 GB card ends with 5.1 GiB of cache and 0.35 GiB free, so
 * about 6.9 GiB goes to everything else; a second card in a layer split about 2 GiB).
 */
export const FIRST_CARD_OTHER_GB = 6.9;
export const NEXT_CARD_OTHER_GB = 2.0;
/**
 * `--batch-mtp` lets each slot verify draft tokens. Measured on a 12 GB RTX 3060 (docs/BATCHING.md says it pays on larger
 * cards): a lone chat 35.6 tok/s with it against 37.8 without, 2 chats 37.2 against 33.8 together, 4 chats 42.5 against 38.4 -
 * a small gain for several chats, a small loss for one, and 400 MiB less expert cache. Off until a card shows it pays; Expert
 * mode adds it by hand (`--batch-mtp` in the engine arguments).
 */
export const BATCH_MTP_POLICY = 'off';

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
  // Without a hardware read nothing can be sized: one at a time is the safe default (a number can still be chosen).
  if (!cards.length) return { slots: PARALLEL_POLICY === 'fixed' ? requested : 1, batchMtp: false, explicit: false, requested, note: null, slotGb: slot };
  const total = cards.reduce((a, b) => a + b, 0);
  const soloCache = Math.max(0, total - FIRST_CARD_OTHER_GB - NEXT_CARD_OTHER_GB * (cards.length - 1));
  let slots = requested;
  if (PARALLEL_POLICY === 'fit') while (slots > 1 && slots * slot > CACHE_LOSS_SHARE * soloCache) slots -= 1;
  const note = slots >= requested ? null
    : slots > 1 ? `Your graphics card has room for ${slots} requests at once with this context size, so Strata set that. A smaller context size leaves room for more.`
      : 'Strata set this model to one request at a time: your graphics card is best used that way, because more at once would make every answer slower. Helper agents and a second chat take their turn instead. A smaller context size leaves room for more at once.';
  return { slots, batchMtp: BATCH_MTP_POLICY === 'on' && slots > 1 && facts.mtp && !facts.batchMtp && facts.gpus.length === 1, explicit: false, requested, note, slotGb: slot };
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
