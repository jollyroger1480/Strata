# Agent bridge (Strata chats as DeepSeek Harness sessions)

The Harness is a separate, pinned component (`DSH 0.2.1-alpha.3`, commit `57190202`, MIT). The host finds it through
`STRATA_HARNESS_DIR` (or `<STRATA_COMPONENTS_DIR>/harness`), starts one `dsh --profile web` process on a loopback port,
and talks to its Gateway. Nothing here downloads or builds anything.

| File | Job |
| --- | --- |
| `harness-runtime.mjs` | locate the component, start/stop the process, scrub its environment, clean up a leftover process from a dead host |
| `gateway-client.mjs` | HTTP RPC plus the `remote.mux` WebSocket (`$events`, `session/follow`) |
| `profile.mjs` | generate the Harness profile: one route per ready local instance and per saved API connection; credential names only |
| `events.mjs` | pure projection of Harness session events into the chat's NDJSON frames |
| `children.mjs` | pure projection of one helper agent's events into its row and its approval card |
| `preset.mjs` | the agent preset (which tools the chat has) |
| `plugins/strata-gate.mjs` | the Manual-mode approval gate (a Harness plugin, loaded by path) |
| `agent-bridge.mjs` | the chat-to-session map, turns, approvals, Stop, status |

## Frames sent on `/api/chat/stream` for an agent turn

`queued`, `start` (`agent:true`), `reasoning`, `delta`, `agent-tool` (new: upserts a tool card by call id with
`running | proposed | done | denied | error`), `done` (usage, `finishReason`, `cancelled`), `error`, and `notice`
(why a chat is not using the agent). The chat UI renders them with its existing components.

## Routes

- `GET /api/agent/status`: component, routes, active turns, pending approvals (no tokens, no keys)
- `POST /api/chat/approval {sessionId, toolId, decision: approve|deny}`
- `POST /api/chat/stop {sessionId}`: cancels that chat's Harness session and answers once the turn has ended
- `GET /api/agent/pending?chatId=`

## Behaviour worth knowing

- **Stop is per chat.** `session/cancel` is only an admission; the bridge waits for the durable `turn/end`. Stop before the
  prompt reached the Harness cancels the start. The process is never killed to implement Stop.
- **A closed tab is not Stop.** The turn keeps running; an approval it needs stays pending until someone answers.
- **Resume.** The chat id maps to a Harness session id (`<data>/agent/sessions.json`). Reopening prompts that session; the
  Harness resumes it from its own record, so nothing is replayed. (`session/create` with a started id returns a new
  session, so it is never used for resume.)
- **Models.** Each chat sends its choice; the bridge calls `session/selectModel` for that route before the prompt. A route
  that is gone is a typed `model-unavailable` error, never another model. Keys travel as environment variables named by
  `apiKeyEnv`; the agent's shell does not inherit them, and logs are redacted.
- **No answerer, no hang.** The agent's own questions (`ask_user_question`) are declared unanswerable until the app has a
  surface for them.
- **Fallback.** Without the component the host says so (`/api/agent/status`, a `notice` frame) and the chat runs on the
  direct path with the model the user picked.

## Modes (the Mode menu)

The chat's Mode menu is real Harness behaviour. Each turn the page sends the chat's mode (`mode` in the `/api/chat/stream` body); before the prompt reaches the Harness the bridge puts that chat's session into the matching permission preset with the Harness's own `/permission` command (`commands/execute`), reads the `permissions` projection back, and refuses to start the turn (typed error `mode-unavailable`) if it does not match. The preset table is ours (profile entry `permission`).

| Menu | Preset | Sandbox | Approval | What it means |
| --- | --- | --- | --- | --- |
| Manual (default) | `strata-manual` | open (`danger-full-access`) | `ask` | The Strata gate (`plugins/strata-gate.mjs`) asks BEFORE any change runs: one Approve/Deny card, nothing has happened when it appears, no blocked attempt, no sandbox or PowerShell error. Reads and read-only commands run without a question; anything the gate cannot prove read-only asks. |
| Plan | `strata-plan` | `read-only` | `never` | Nothing can change. A request is rejected without asking anyone, so no Approve card is shown; the card ends "Not allowed in this mode". |
| Bypass | `strata-bypass` | `danger-full-access` | `never` | Full access without prompts, as before. |

- Manual relies on the gate, so a Manual turn does not start unless the bridge has confirmed the gate is loaded (`/strata-gate` command): without it Manual would be an open sandbox that nobody asks about.
- The gate listens to `tools/pre-execute` (the seam the Harness's own Auto-review layer uses) and returns an `ask` decision; the Harness's approval service then asks, runs the tool only after `allowed-once`, and records the answer. Read-only tools and a short allowlist of look-only shell commands (Get-ChildItem, git status, ...) run unasked; MCP and unknown tools ask.
- The mode is read once per turn; an unknown or missing value is Manual.

## Tools

`preset.mjs` is the agent preset: the Harness `standard` preset minus the scheduler (Tasks is the only one), web search (needs a key the app has none of; `search: true` switches it on), the agent's own question tool (no place to answer) and the Harness's plan mode. `tests/agent-tools.test.mjs` compares the list with the pinned Harness's own preset file and the tools it actually offers. The model's tool servers (Settings, Tools) are connected as Harness MCP client rows (`mcpRows` in `profile.mjs`); their secrets travel in the agent's environment, never in the profile file.

## Helper agents (subagents)

- A Harness `subagent/catalog` event on the chat's session creates a row; `children.mjs` folds the helper's `session/follow` (address `{kind:'subagent', parentSessionId, childSessionId}`) into `{type:'agent', agent:{id,name,description,status,durationMs,tokens,updatedAt}}`, the shape the Subagents popover already reads. The delegation tools have no card.
- A helper's request for approval is the same card (state `proposed`, id `<helper session>:<call>`), answered with the same `/api/chat/approval`; a chat can only answer its own helpers.
- The run (the chat's one NDJSON response) stays open until the parent is idle, every helper has finished and reported back, and no armed goal is waiting; `done` is sent then.
- Stop: pauses an active goal, interrupts every helper (`subagents/interruptByParent`; `session/cancel` on a helper session is refused by the Harness), cancels the parent, waits; a report that would wake the parent afterwards is cancelled too.
- The number of helpers at once equals the engine's request slots (`maxActiveSubagents` in the profile); the Harness refuses more with a message the model reads.

## /goal

`/goal` typed in a chat goes to `bridge.goalCommand`: nothing = show the goal, `clear`, `pause`, `resume`, `edit <text>`, `complete`, anything else = a new goal (one unfinished goal at a time). Create/resume use the Harness goal Remotes (`goals/create|get|resume|pause|edit|clear|complete`); the Harness's goal driver runs the rounds and they stream into the chat ("Working on the goal, step N"); the run ends when the goal stops (complete, paused, blocked, round limit - the default cap is the Harness's 256, a run with `maxRounds` is capped) and says how in plain words. After a restart an active goal stays disarmed until `/goal resume`.

## Plain words

Failures the person sees are plain sentences with a next step (`AgentUnavailable.message`); the technical text is in `.details` / the `details` field of error and notice frames, and in `/api/agent/status` `detail`. A failed tool's card result is a plain sentence followed by "Details: <original text>".

## Tests

`tests/agent-events.test.mjs` (pure), `tests/agent-bridge.test.mjs`, `tests/agent-modes.test.mjs` and `tests/agent-host.test.mjs` (need a built Harness:
`STRATA_HARNESS_DIR=<checkout>`, otherwise they skip and say so). The model in those tests is `tests/helpers/scripted-worker.mjs`.
`scripts/agent-flow.mjs` drives the real app in Chrome and takes screenshots.
