/**
 * A scripted OpenAI-compatible chat-completions worker, for the agent-bridge tests.
 *
 * It stands in for the Strata API worker so that the Harness agent loop, the tool executor, the
 * approval gate and the host bridge are all real while the model is scripted. A scripted model proves
 * the integration path, not model quality; the real-model run is a separate step.
 *
 * Script selection is by a directive in the last user message (`SCRIPT:<name> <args>`), never by
 * request count: the Harness makes requests a script author does not plan for (a session title, a
 * re-ask after a tool result), and a counter would shift every later reply.
 *
 *   SCRIPT:hello                    reasoning then a short text reply
 *   SCRIPT:slow [n]                 n (default 400) text chunks, 40 ms apart: a window for Stop
 *   SCRIPT:tool <file> <text>       one shell call that appends <text> to <file>; it asks the sandbox for
 *                                   wider access, so the approval gate is exercised
 *   SCRIPT:plain <file> <text>      the same call without the escalation (no approval needed)
 *   SCRIPT:echo <text>              replies with <text>
 *   SCRIPT:ask                      one ask_user_question call (the agent asking the person something)
 *   SCRIPT:envdump <file>           one shell call that writes the shell's environment to <file>
 *   SCRIPT:retrywrite <file> <text> the `write` tool, like a careful model: when the sandbox denies it and offers
 *                                   an escalation, retry once with `sandbox_permissions: workspace-write`
 *   SCRIPT:retryshell <file> <text> the same with the shell tool (an in-workspace append)
 *   SCRIPT:read <file>              the `read` tool on <file>
 *
 * A request whose last message is a tool result is answered with a text that quotes the result.
 */
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';

const delay = ms => new Promise(r => setTimeout(r, ms));

export async function startScriptedWorker({ port = 0, chunkDelayMs = 40, model = 'scripted-model' } = {}) {
  const requests = [];
  let aborted = 0;
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    const json = (code, body) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(body)); };
    if (req.method === 'GET' && url.pathname === '/health') return json(200, { ok: true });
    if (req.method === 'GET' && url.pathname === '/v1/models') return json(200, { object: 'list', data: [{ id: model, object: 'model' }] });
    if (req.method === 'GET' && url.pathname === '/__requests') return json(200, { requests, aborted });
    if (req.method !== 'POST' || !url.pathname.endsWith('/chat/completions')) return json(404, { error: 'no route' });
    let raw = '';
    for await (const c of req) raw += c;
    const body = JSON.parse(raw);
    const messages = body.messages ?? [];
    const last = messages[messages.length - 1] ?? {};
    const text = m => typeof m?.content === 'string' ? m.content : Array.isArray(m?.content) ? m.content.map(p => p.text ?? '').join(' ') : '';
    // The Harness appends runtime-context messages after the prompt, so the newest user message is not
    // the prompt: take the newest one that carries a directive.
    const directive = [...messages].reverse().filter(m => m.role === 'user')
      .map(m => /SCRIPT:(\w+)([^\n]*)/.exec(text(m))).find(Boolean);
    requests.push({
      at: Date.now(), model: body.model, stream: body.stream === true,
      toolNames: (body.tools ?? []).map(t => t.function?.name),
      lastRole: last.role, directive: directive?.[1] ?? null, messageCount: messages.length,
      authorization: req.headers.authorization ? 'present' : 'absent',
      authSha: req.headers.authorization ? createHash('sha256').update(req.headers.authorization).digest('hex') : null,
      promptChars: raw.length,
    });
    const shellName = () => (body.tools ?? []).find(t => ['bash', 'pwsh'].includes(t.function?.name))?.function?.name ?? 'bash';
    const shellCommand = (file, value) => (shellName() === 'pwsh' ? `Add-Content -Path '${file}' -Value '${value}'` : `echo ${value} >> ${file}`);
    let closed = false;
    res.on('close', () => { if (!res.writableEnded) { closed = true; aborted++; } });

    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    const id = `chatcmpl-${requests.length}`;
    const send = (delta, finish = null, extra = {}) => res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: 0, model: body.model, choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}\n\n`);
    const finish = (reason, tokens = 10) => {
      send({}, reason);
      res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: 0, model: body.model, choices: [], usage: { prompt_tokens: 100 + messages.length, completion_tokens: tokens, total_tokens: 100 + messages.length + tokens } })}\n\n`);
      res.write('data: [DONE]\n\n'); res.end();
    };
    const toolCall = async (name, call) => {
      send({ role: 'assistant', content: '', reasoning_content: `I will use ${name}.` });
      await delay(chunkDelayMs);
      const argsJson = JSON.stringify(call);
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name, arguments: '' } }] });
      for (let i = 0; i < argsJson.length; i += 17) { send({ tool_calls: [{ index: 0, function: { arguments: argsJson.slice(i, i + 17) } }] }); await delay(5); }
      return finish('tool_calls', 30);
    };
    const say = async (parts, tokens) => {
      send({ role: 'assistant', content: '' });
      for (const p of parts) { if (closed) return; send({ content: p }); await delay(chunkDelayMs); }
      if (!closed) finish('stop', tokens ?? parts.length);
    };

    // A model that read "[sandbox: escalation available ...]" retries the exact call once with the narrowest wider mode.
    // Only this prompt's own tool results count: a chat that has run tools before carries older ones.
    const promptAt = messages.findLastIndex(m => m.role === 'user' && /SCRIPT:/.test(text(m)));
    const toolMessages = messages.slice(promptAt + 1).filter(m => m.role === 'tool');
    if (last.role === 'tool' && ['retrywrite', 'retryshell'].includes(directive?.[1]) && toolMessages.length === 1 && /escalation available/.test(text(last))) {
      const [file, ...rest] = (directive[2] ?? '').trim().split(/\s+/);
      const value = rest.join(' ') || 'run';
      return toolCall(directive[1] === 'retrywrite' ? 'write' : shellName(), directive[1] === 'retrywrite'
        ? { file_path: file, content: `${value}
`, sandbox_permissions: 'workspace-write', justification: 'The user asked for this file to be written.' }
        : { command: shellCommand(file, value), description: `Append one line to ${file}`, sandbox_permissions: 'workspace-write', justification: 'The user asked for this file to be written.' });
    }
    if (last.role === 'tool') {
      const result = text(last).replace(/\s+/g, ' ').slice(0, 120);
      return say(['Tool finished. ', `Result: ${result}`]);
    }
    const [name, args] = [directive?.[1], (directive?.[2] ?? '').trim()];
    // The Harness's session-title request carries the same user text but offers no tools.
    if (!(body.tools ?? []).length && messages.some(m => /concise title/i.test(text(m)))) return say(['Scripted title']);
    if (name === 'slow') {
      const n = Number(args) || 400;
      return say(Array.from({ length: n }, (_, i) => `w${i} `));
    }
    if (name === 'echo') return say([args]);
    if (name === 'retrywrite' || name === 'retryshell') {
      const [file, ...rest] = args.split(/\s+/);
      const value = rest.join(' ') || 'run';
      return name === 'retrywrite'
        ? toolCall('write', { file_path: file, content: `${value}
` })
        : toolCall(shellName(), { command: shellCommand(file, value), description: `Append one line to ${file}` });
    }
    if (name === 'read') return toolCall('read', { file_path: args });
    if (name === 'ask') {
      send({ role: 'assistant', content: '' });
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name: 'ask_user_question', arguments: JSON.stringify({ questions: [{ id: 'q1', question: 'Which colour?', options: [{ label: 'Red' }, { label: 'Blue' }] }] }) } }] });
      return finish('tool_calls', 20);
    }
    if (name === 'tool' || name === 'plain' || name === 'envdump') {
      const [file, ...rest] = args.split(/\s+/);
      const shell = (body.tools ?? []).find(t => ['bash', 'pwsh'].includes(t.function?.name))?.function?.name ?? 'bash';
      const value = rest.join(' ') || 'run';
      let command = shell === 'pwsh' ? `Add-Content -Path '${file}' -Value '${value}'` : `echo ${value} >> ${file}`;
      if (name === 'envdump') command = shell === 'pwsh' ? `Get-ChildItem Env: | ForEach-Object { $_.Name + '=' + $_.Value } | Set-Content -Path '${file}'` : `env > ${file}`;
      const call = { command, description: `Append one line to ${file}` };
      if (name === 'tool') {
        call.sandbox_permissions = 'danger-full-access';
        call.justification = 'The write must not be refused by the file sandbox.';
      }
      send({ role: 'assistant', content: '', reasoning_content: 'I will write the file with the shell.' });
      await delay(chunkDelayMs);
      const argsJson = JSON.stringify(call);
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name: shell, arguments: '' } }] });
      // Fragment the arguments the way real servers do.
      for (let i = 0; i < argsJson.length; i += 17) {
        send({ tool_calls: [{ index: 0, function: { arguments: argsJson.slice(i, i + 17) } }] });
        await delay(5);
      }
      return finish('tool_calls', 30);
    }
    // hello, and the default for anything unscripted (including the Harness's own title request)
    send({ role: 'assistant', content: '', reasoning_content: 'Considering a greeting.' });
    await delay(chunkDelayMs);
    return say(['Hello ', 'from ', 'the ', 'scripted ', 'worker.']);
  });
  await new Promise(r => server.listen(port, '127.0.0.1', r));
  return {
    port: server.address().port,
    url: `http://127.0.0.1:${server.address().port}/v1`,
    requests, get aborted() { return aborted; },
    close: () => new Promise(r => { server.closeAllConnections?.(); server.close(r); }),
  };
}

if (import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}` || process.argv[1]?.endsWith('scripted-worker.mjs')) {
  const w = await startScriptedWorker({ port: Number(process.env.PORT ?? 0) });
  console.log(`scripted worker on ${w.url}`);
}
