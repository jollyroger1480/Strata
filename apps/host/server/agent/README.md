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

## Tests

`tests/agent-events.test.mjs` (pure), `tests/agent-bridge.test.mjs` and `tests/agent-host.test.mjs` (need a built Harness:
`STRATA_HARNESS_DIR=<checkout>`, otherwise they skip and say so). The model in those tests is `tests/helpers/scripted-worker.mjs`.
`scripts/agent-flow.mjs` drives the real app in Chrome and takes screenshots.
