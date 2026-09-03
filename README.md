# research-committee

## What this is

A small multi-agent research assistant — a "committee of agents" — built to
learn the **A2A (Agent2Agent) protocol** properly, end-to-end, rather than
just reading about it.

A human asks a research question. Four independent agents collaborate over
A2A to answer it: `lit_search` finds sources, `synthesis` drafts an answer,
`critic` checks the draft's claims against those sources, and `chair`
orchestrates the other three and talks to the human. Each agent is its own
process, on its own port, speaking A2A's JSON-RPC/SSE protocol — nothing is
imported or called directly between them.

## Why A2A (briefly)

A2A and MCP solve different problems. MCP is agent-to-*tool*: a model calls
a well-known function with a defined schema. A2A is agent-to-*agent*: two
independent, possibly differently-implemented agents negotiate a
multi-turn task, and neither has to know anything about the other's
internals. Agents in this repo are deliberately treated as opaque peers —
`chair` never imports `synthesis`'s code, it only knows `synthesis` exists,
what it can do, and how to talk to it because `synthesis` publishes an
**Agent Card** describing exactly that.

For the protocol itself — Agent Cards, Tasks, `message/stream`, artifacts,
task states — see the spec: https://a2a-protocol.org. This README only
covers how this repo uses it.

## Architecture

```
                         human
                           │
                           │ message/stream ("research-question")
                           ▼
                   ┌───────────────┐
                   │     chair     │  :9000
                   │ (orchestrator)│
                   └───┬───┬───┬───┘
           ┌───────────┘   │   └───────────┐
           ▼                ▼               ▼
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │   lit_search   │ │   synthesis   │ │    critic     │
   │     :9001      │ │     :9002     │ │     :9003     │
   │ find-sources   │ │ draft-answer  │ │ verify-claims │
   └────────────────┘ └───────────────┘ └───────────────┘
```

Request flow for one research question:

1. `chair` receives the question from the human and generates one
   `contextId` for the whole exchange.
2. `chair` calls `lit_search` (streamed), relaying each found source
   upward as its own status updates.
3. `chair` feeds the collected sources into `synthesis` (streamed),
   relaying each drafted chunk upward the same way.
4. `chair` feeds the finished draft + sources into `critic` (single call).
5. `chair` returns one final artifact: the answer plus `critic`'s verdicts.

Every one of those four calls is a **separate A2A Task** — a
`lit_search` task, a `synthesis` task, a `critic` task, and `chair`'s own
task to the human — but they all carry the same `contextId`, so the whole
exchange is traceable as one logical conversation even though there's no
single shared Task spanning it.

## The agents

### `lit_search` (9001)

Streams candidate sources one at a time — a separate
`TaskArtifactUpdateEvent` per source (`append`/`lastChunk` sequencing),
not one batch dump — so a caller sees results arrive incrementally
regardless of how the underlying search actually behaves.

Search itself sits behind a swappable `SearchProvider` adapter
(`agents/lit_search/search_provider.py`): `FakeSearchProvider` (default,
deterministic, no API key) or `TavilySearchProvider` (real web search).
The executor's streaming/task logic doesn't know or care which one is
behind it — the A2A contract never depends on the search API.

### `synthesis` (9002)

Streams a drafted answer as incremental `TextPart` chunks (`append=true`,
final chunk `lastChunk=true`) rather than returning the whole draft at
once.

It also implements a real `input-required` pause: before drafting, it
checks whether the sources are too thin (`<2`) or contradictory (opposing
sentiment keywords like "increases" vs. "decreases"). The contradiction
check is gated behind a **topic-relatedness threshold**
(`TOPIC_RELATEDNESS_THRESHOLD`) — worth calling out specifically, because
it was a real bug caught during verification: without the gate, two
*unrelated* sources that each happened to contain one opposite-sentiment
keyword (e.g. "increases" describing solar panel efficiency, "decreases"
describing recycling costs) would falsely trigger a pause. The threshold
requires the two sources to also share enough vocabulary to plausibly be
about the same thing before treating opposite sentiment as a genuine
disagreement.

### `critic` (9003)

Single-shot, non-streaming: one `working` status, one final `DataPart`
artifact with a per-claim verdict list, then `completed`. Claim
verification is a deliberately naive keyword-overlap heuristic against a
named `SUPPORT_THRESHOLD` constant — explicitly flagged in
`agents/critic/matching.py` as a placeholder for real fact-checking, not
a finished feature.

### `chair` (9000)

The only agent a human talks to. On startup it fetches and validates all
three downstream Agent Cards (reachable + advertising the expected skill)
*before* accepting any request — if one is missing or misconfigured, it
fails loudly at startup with a clear log line, rather than discovering
the problem mid-conversation.

It also handles the hardest case in this repo: **two-hop `input-required`
propagation**. When `synthesis` pauses, `chair` doesn't guess or auto-fill
an answer — it pauses its own task too, forwards `synthesis`'s exact
question up to the human unmodified, and waits. When the human replies,
`chair` resumes and forwards that reply to the correct `synthesis`
sub-task (recovered from `chair`'s own `Task.metadata`, stashed there for
exactly this purpose) to continue where it left off.

## Why `input-required` matters

Most tutorial-level A2A implementations skip `input-required` entirely —
it's easy to build a one-shot "send request, stream response" flow and
call it done. But a protocol that can't pause mid-task, ask a clarifying
question, and resume from a caller's reply is just a fancy RPC call, not
a collaboration protocol. `synthesis` and `chair` are the two places in
this repo where that round trip is implemented for real, including the
harder version — a pause that has to propagate up through an
intermediary (`chair`) rather than terminating at the immediate caller.

## Running it locally

```bash
git clone <this-repo>
cd research-committee-A2A
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Start the three sub-agents, then `chair` (it validates the other three at
startup, so they need to be up first):

```bash
python -m agents.lit_search &
python -m agents.synthesis &
python -m agents.critic &
python -m chair &
```

Fetch an Agent Card directly:

```bash
python -m common.debug_client --url http://localhost:9001 --text "check the card"
```

Drive a full request through `chair`:

```bash
python -m common.debug_client --url http://localhost:9000 \
  --text "What does the evidence say about sleep and memory consolidation?"
```

`debug_client.py` also supports DataPart payloads (`--data`/`--data-file`,
for hitting `synthesis`/`critic` directly) and replying to a paused task
(`--task-id`/`--context-id`) — see its own `--help` and docstring.

### Fake vs. live search

`lit_search` defaults to `FakeSearchProvider` — no key needed, safe for
repeated `debug_client.py` runs. To use real search:

```bash
LIT_SEARCH_PROVIDER=tavily TAVILY_API_KEY=<your key> python -m agents.lit_search
```

`TAVILY_API_KEY` is read from the environment only — never hardcoded, and
if `LIT_SEARCH_PROVIDER=tavily` is set without a key, `lit_search` fails
loudly at startup instead of on the first search.

## Current limitations / known TODOs

- **No auth or Agent Card signature verification** — flagged as a `v1`
  TODO in every agent's executor. Fine for localhost; not for anything
  beyond it.
- **`critic`'s matching is naive keyword overlap**, not real NLP or an
  LLM call — explicitly a placeholder (see `agents/critic/matching.py`).
- **Live Tavily response-shape verification is outstanding.** The
  provider adapter's request/response/error handling was verified
  against a stub server and one genuine live network failure, but a
  real-key smoke test against the live Tavily API hasn't been run in
  this repo's development sandbox (egress was restricted there) — see
  `SPEC.md`'s Step 5 section for the details.
- **Localhost-only, no persistent storage, no multi-tenancy, no
  gRPC/REST bindings** — all explicitly out of scope for `v1` (see
  `SPEC.md`).

## Build history

`SPEC.md` is the original design doc this was built from, step by step
(fake `lit_search` → debug client → `synthesis`/`critic` → `chair` →
real search). Worth reading if you want the reasoning behind each piece,
not just the result.
