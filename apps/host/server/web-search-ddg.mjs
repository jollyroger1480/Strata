/**
 * Web search with no account and no key: DuckDuckGo's own web-page search results.
 *
 * The Harness has three search backends and all of them need a paid service key. This is the fourth: it asks DuckDuckGo's plain
 * HTML search page the way a browser does and reads the result list out of the page. It is the default for the agent's
 * `web_search` tool; the person can switch to a keyed service in Settings -> Tools.
 *
 * Being polite is part of the design:
 *   - an ordinary browser User-Agent and the page's own request shape (nothing hidden, nothing faked beyond the User-Agent);
 *   - at most one request every 1.5 s and 20 a minute from this program, with the same query answered from a 10 minute cache;
 *   - a 15 s time limit, no retries in a loop; when DuckDuckGo asks for a human check or refuses, the answer says so in plain
 *     words and the model is told to carry on without search.
 * Results come from DuckDuckGo; they are not ranked or filtered here.
 *
 * Nothing in this file touches the Harness, so it can be tested alone. `strata-search.mjs` is the plugin that registers it.
 */

export const DDG_PROVIDER_ID = 'strata-duckduckgo';

const HTML_URL = 'https://html.duckduckgo.com/html/';
const LITE_URL = 'https://lite.duckduckgo.com/lite/';
const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';
const CACHE_MS = 10 * 60 * 1000;
const MIN_GAP_MS = 1500;
const PER_MINUTE = 20;

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'", '#x27': "'", '#x2F': '/' };
const decode = text => String(text).replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (whole, key) => {
  if (key[0] === '#') {
    const code = key[1].toLowerCase() === 'x' ? parseInt(key.slice(2), 16) : parseInt(key.slice(1), 10);
    return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : whole;
  }
  return ENTITIES[key.toLowerCase()] ?? whole;
});
const plain = html => decode(String(html).replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();

/** DuckDuckGo wraps every result link in its own redirect; the real address is in `uddg`. */
export function realUrl(href) {
  let value = decode(href).trim();
  if (value.startsWith('//')) value = `https:${value}`;
  try {
    const url = new URL(value, 'https://duckduckgo.com/');
    const target = url.searchParams.get('uddg');
    if (target) return new URL(target).href;
    if (url.hostname.endsWith('duckduckgo.com')) return null;      // an internal link, an ad or a "more results" link
    return /^https?:$/.test(url.protocol) ? url.href : null;
  } catch { return null; }
}

/** The result list of the HTML page. Tolerant: a result needs a link and a title; the snippet is optional. */
export function parseHtmlResults(html) {
  const out = [];
  const links = [...String(html).matchAll(/<a\b[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>/gi)];
  for (let i = 0; i < links.length; i++) {
    const open = links[i];
    const start = open.index + open[0].length;
    const next = links[i + 1]?.index ?? html.length;
    const chunk = html.slice(start, next);
    const close = chunk.indexOf('</a>');
    if (close < 0) continue;
    const href = /href="([^"]*)"/i.exec(open[0])?.[1];
    const url = href ? realUrl(href) : null;
    const title = plain(chunk.slice(0, close));
    if (!url || !title) continue;
    // An ad is a result block whose own opening tag says so (the block that holds this link, not the one before it).
    const block = html.lastIndexOf('<div class="result', open.index);
    if (block >= 0 && /result--ad|result__ad/.test(html.slice(block, html.indexOf('>', block) + 1))) continue;
    const snippetMatch = /<(?:a|td|div)\b[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>([\s\S]*?)<\/(?:a|td|div)>/i.exec(chunk);
    const snippet = snippetMatch ? plain(snippetMatch[1]) : '';
    out.push({ url, title, ...(snippet ? { snippet } : {}) });
  }
  return dedupe(out);
}

/** The result list of the lite page (a table), used when the HTML page gives nothing. */
export function parseLiteResults(html) {
  const out = [];
  const links = [...String(html).matchAll(/<a\b[^>]*class=['"][^'"]*\bresult-link\b[^'"]*['"][^>]*>/gi)];
  for (let i = 0; i < links.length; i++) {
    const open = links[i];
    const start = open.index + open[0].length;
    const next = links[i + 1]?.index ?? html.length;
    const chunk = html.slice(start, next);
    const close = chunk.indexOf('</a>');
    if (close < 0) continue;
    const href = /href=['"]([^'"]*)['"]/i.exec(open[0])?.[1];
    const url = href ? realUrl(href) : null;
    const title = plain(chunk.slice(0, close));
    if (!url || !title) continue;
    const snippetMatch = /<td\b[^>]*class=['"][^'"]*\bresult-snippet\b[^'"]*['"][^>]*>([\s\S]*?)<\/td>/i.exec(chunk);
    const snippet = snippetMatch ? plain(snippetMatch[1]) : '';
    out.push({ url, title, ...(snippet ? { snippet } : {}) });
  }
  return dedupe(out);
}

function dedupe(rows) {
  const seen = new Set();
  return rows.filter(r => !seen.has(r.url) && seen.add(r.url));
}

/** DuckDuckGo asks a program to prove it is a person by serving a challenge page instead of results. */
export function looksBlocked(status, html) {
  if (status === 202 || status === 403 || status === 418 || status === 429) return true;
  return /anomaly-modal|anomaly\.js|unusual traffic|are you a (?:robot|human)|captcha|challenge-form/i.test(String(html).slice(0, 60000))
    && !/class="[^"]*\bresult__a\b/.test(html);
}

export class SearchUnavailable extends Error {
  constructor(message, code = 'WEB_PROVIDER_ERROR', options) { super(message, options); this.name = 'WebError'; this.code = code; }
}

export const BLOCKED_MESSAGE = 'DuckDuckGo is limiting searches from this computer right now. Try again in a few minutes, or add a search key in Settings, Tools.';

/**
 * @param {{fetch?: typeof fetch, now?: () => number, sleep?: (ms: number) => Promise<void>}} [deps]
 */
export function createDuckDuckGoSearch({ fetch: fetchImpl = globalThis.fetch, now = Date.now, sleep = ms => new Promise(r => setTimeout(r, ms)) } = {}) {
  const cache = new Map();
  const starts = [];
  let last = 0;
  let chain = Promise.resolve();

  /** One request at a time, spaced out, and at most PER_MINUTE a minute. */
  function turn(signal) {
    const run = chain.then(async () => {
      for (;;) {
        signal?.throwIfAborted();
        const t = now();
        while (starts.length && t - starts[0] > 60_000) starts.shift();
        const wait = Math.max(MIN_GAP_MS - (t - last), starts.length >= PER_MINUTE ? 60_000 - (t - starts[0]) : 0);
        if (wait <= 0) break;
        if (wait > 30_000) throw new SearchUnavailable('Too many searches in a short time. Try again in a minute.', 'WEB_RATE_LIMITED');
        await sleep(wait);
      }
      last = now();
      starts.push(last);
    });
    chain = run.catch(() => {});
    return run;
  }

  async function ask(url, query, signal) {
    const timeout = AbortSignal.timeout(15_000);
    const res = await fetchImpl(url, {
      method: 'POST', redirect: 'follow',
      headers: { 'user-agent': USER_AGENT, 'content-type': 'application/x-www-form-urlencoded', accept: 'text/html,application/xhtml+xml', 'accept-language': 'en-US,en;q=0.8' },
      body: new URLSearchParams({ q: query, b: '' }).toString(),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    const html = await res.text();
    return { status: res.status, html };
  }

  return {
    id: DDG_PROVIDER_ID,
    available: () => typeof fetchImpl === 'function',
    async search(request, signal) {
      const query = String(request.query ?? '').trim().slice(0, 400);
      if (!query) throw new SearchUnavailable('Type what to search for.', 'WEB_PROVIDER_ERROR');
      const key = query.toLowerCase();
      const hit = cache.get(key);
      if (hit && now() - hit.at < CACHE_MS) return { sources: hit.sources.slice(0, request.maxResults ?? 10), truncated: false };
      let sources = [];
      let blocked = false;
      for (const [url, parse] of [[HTML_URL, parseHtmlResults], [LITE_URL, parseLiteResults]]) {
        await turn(signal);
        let page;
        try { page = await ask(url, query, signal); }
        catch (error) {
          if (signal?.aborted) throw new SearchUnavailable('The search was cancelled.', 'WEB_ABORTED', { cause: error });
          if (error?.name === 'TimeoutError') throw new SearchUnavailable('The search took too long. Try again.', 'WEB_TIMEOUT', { cause: error });
          throw new SearchUnavailable('The search could not reach DuckDuckGo. Check the internet connection and try again.', 'WEB_PROVIDER_ERROR', { cause: error });
        }
        if (looksBlocked(page.status, page.html)) { blocked = true; continue; }
        sources = parse(page.html);
        if (sources.length) { blocked = false; break; }
      }
      if (!sources.length) {
        if (blocked) throw new SearchUnavailable(BLOCKED_MESSAGE, 'WEB_RATE_LIMITED');
        return { sources: [], truncated: false };          // a real search with no results is an answer, not an error
      }
      cache.set(key, { at: now(), sources });
      if (cache.size > 60) cache.delete(cache.keys().next().value);
      return { sources: sources.slice(0, request.maxResults ?? 10), truncated: false };
    },
  };
}
