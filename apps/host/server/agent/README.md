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

The chat's Mode menu is real Harness behaviour. Each turn the page sends the chat's mode (`mode` in the `/api/chat/stream`
body); before the prompt reaches the Harness the bridge puts that chat's session into the matching permission preset with the
Harness's own `/permission` command (`commands/execute`), reads the session's `permissions` projection back, and refuses to
start the turn (typed error `mode-unavailable`) if it does not match. The preset table is ours (profile entry `permission`);
the Harness's own `read-only`, `workspace-write` and `danger-full-access` presets are not offered.

| Menu | Preset | Harness sandbox | Approval policy | What it means |
| --- | --- | --- | --- | --- |
| Manual (default) | `strata-manual` | `read-only` | `ask` | The sandbox denies every file change, so the agent must request a one-shot widening (`sandbox_permissions`), and that request is the Approve/Deny card. Reads and read-only commands run without a question. |
| Plan | `strata-plan` | `read-only` | `never` | Nothing can change. A request to widen is rejected without asking anyone, so no Approve card is ever shown; the card ends "Not allowed in this mode". The agent reads, plans and answers. |
| Bypass | `strata-bypass` | `danger-full-access` | `never` | The Harness's own full-access, never-ask bundle. Writes and commands anywhere on this computer run without a question. |

- The mode is read once per turn. Changing the menu during a turn affects the next turn; the running one keeps the permissions it started with.
- An unknown or missing value (an older stored `auto` or `accept-edits`) is Manual, never something looser.
- The session keeps its preset across host restarts (it is a durable event in the Harness record); the page's choice is stored with the chat.
- Bypass is the only full-access mode (the pinned Harness's `danger-full-access` preset: sandbox `danger-full-access`, approval `never`). Manual and Plan stay read-only.
- Windows: the read-only shell runs PowerShell in Constrained Language Mode. A shell command can fail inside Manual or Plan for reasons that have nothing to do with approvals (the card shows the PowerShell error).

## Tests

`tests/agent-events.test.mjs` (pure), `tests/agent-bridge.test.mjs`, `tests/agent-modes.test.mjs` and `tests/agent-host.test.mjs` (need a built Harness:
`STRATA_HARNESS_DIR=<checkout>`, otherwise they skip and say so). The model in those tests is `tests/helpers/scripted-worker.mjs`.
`scripts/agent-flow.mjs` drives the real app in Chrome and takes screenshots.
