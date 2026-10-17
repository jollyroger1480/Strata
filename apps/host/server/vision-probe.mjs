/**
 * Does a model server on this network read pictures? Asked of its own model list (`/models`, `architecture.input_modalities`), cached for a
 * few minutes. Unknown or unreachable counts as "no": a picture sent to a server without a vision encoder makes the whole request fail.
 */
const TTL_MS = 5 * 60_000;

export function createVisionProbe({ fetcher = fetch, now = Date.now } = {}) {
  const cache = new Map();   // key -> { at, value }
  const keyOf = (conn, modelId) => `${conn.id}|${modelId}`;
  return {
    /** Last known answer for a model (true / false), or undefined when never asked. */
    known: (conn, modelId) => { const e = cache.get(keyOf(conn, modelId)); return e ? e.value : undefined; },
    /** Ask now (at most once per TTL per model). */
    async check(conn, modelId) {
      const k = keyOf(conn, modelId); const hit = cache.get(k);
      if (hit && now() - hit.at < TTL_MS) return hit.value;
      let value = false;
      try {
        const res = await fetcher(String(conn.baseUrl).replace(/\/+$/, '') + '/models', { headers: conn.apiKey ? { authorization: `Bearer ${conn.apiKey}` } : {}, signal: AbortSignal.timeout(4000) });
        if (res.ok) {
          const list = (await res.json())?.data ?? [];
          const row = list.find(m => m?.id === modelId) ?? (list.length === 1 ? list[0] : null);
          const modes = row?.architecture?.input_modalities ?? row?.input_modalities ?? [];
          value = Array.isArray(modes) && modes.includes('image');
        }
      } catch { value = false; }
      cache.set(k, { at: now(), value });
      return value;
    },
  };
}
