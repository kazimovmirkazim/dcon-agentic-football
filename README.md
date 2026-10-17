# DCON — AWS Agentic Football Cup (Baku, 2026)

Five autonomous football agents (GK, DEF, MID, 2×FWD) built for the **AWS Agentic Football Cup**, deployed on
**Amazon Bedrock AgentCore Runtime**. Team **DCON** — players *Titan* (GK), *Reaper* (DEF), *Apex* (MID),
*Ares* and *Zero* (FWD).

Built on top of the official AWS sample:
[aws-samples/sample-ai-possibilities → agentic-football-sample-agents](https://github.com/aws-samples/sample-ai-possibilities)
(MIT-0, see `LICENSE-aws-sample`).

![DCON live match vs Burger bros — real-time coach feed on the right](https://raw.githubusercontent.com/kazimovmirkazim/dcon-agentic-football/main/docs/match-live-coach.jpg)

| Kick-off: DCON (pink) vs Burger bros | Coach "shout" sent to the agents mid-match |
|---|---|
| ![kick-off](https://raw.githubusercontent.com/kazimovmirkazim/dcon-agentic-football/main/docs/match-kickoff.jpg) | ![coach shout](https://raw.githubusercontent.com/kazimovmirkazim/dcon-agentic-football/main/docs/match-coach-shout.jpg) |

**Pre-match fitness:** all 5 agents pass, 25 commands, 100 % active, **97 ms average latency**

![team fitness](https://raw.githubusercontent.com/kazimovmirkazim/dcon-agentic-football/main/docs/team-fitness.jpg)

## Results in numbers

![latency](https://raw.githubusercontent.com/kazimovmirkazim/dcon-agentic-football/main/docs/latency.png)

![command mix](https://raw.githubusercontent.com/kazimovmirkazim/dcon-agentic-football/main/docs/command-mix.png)

## How it works

Every ~2 s the game server sends each agent the full game state (`gameState`, `teamId`, `myPlayers`,
`unlockedCommands`). The agent must answer with one command (`MOVE_TO`, `PASS`, `SHOOT`, `PRESS_BALL`,
`INTERCEPT`, `SLIDE_TACKLE`, `GK_DISTRIBUTE`, …) inside a 5 s budget.

```
game server ──► AgentCore Runtime (1 per position) ──► dcon_core.build_handler
                                                        ├─ normalize_state   (team / id / position formats)
                                                        ├─ tactics()          deterministic layer, ~0 ms
                                                        │    ├─ on the ball: shoot / pass / dribble
                                                        │    ├─ goalkeeper: angle-narrowing shot-stopper
                                                        │    ├─ defending: one presser, line blocking, tackles
                                                        │    └─ attacking shape off the ball
                                                        ├─ ModelChain (Bedrock, time-boxed, fallback chain)
                                                        └─ sanitize / enforce_allowed / fallback (never freeze)
```

### What we learned (iterations v1 → v8)

| Version | Change | Effect |
|---|---|---|
| v1–v3 | LLM decides every tick (Haiku / Sonnet / Nova) | 1–3 s latency, invalid JSON, frozen players |
| v4 | Nova Micro + reflex layer + time-boxed model chain | stable, ~600 ms |
| v6 | team-aware ball holder, no ball-only commands without the ball | fewer wasted ticks |
| v7 | state normalisation (team labels, ids) — fixes "we never have the ball" bug | passes & shots work |
| v8 | fully rule-based decisions, balanced press/pass, accurate shot power | **~100 ms avg latency** (fitness), 0 ms compute |

Key insight: in a real-time game, **latency beats intelligence**. A deterministic tactical layer handles
every clear-cut situation instantly; the LLM chain stays available as an optional layer.

## Engineering deep dive

### 1. The contract with the game server
Each position is a separate **AgentCore Runtime** (a microVM-backed container). The match server calls
`InvokeAgentRuntime` for every player roughly every 2 seconds with a JSON payload:

```json
{"gameState": {"ball": {...}, "players": [...], "score": {...}, "gameTime": 42, "playMode": "OPEN_PLAY",
               "teamChat": [...]},
 "teamId": 0, "myPlayers": [3], "unlockedCommands": ["MOVE_TO","PASS","SHOOT", "..."], "previousCommand": {...}}
```

The agent streams back exactly one command, e.g.
`[{"commandType":"SHOOT","playerId":3,"teamId":0,"parameters":{"aim_location":"BL","power":0.8},"duration":0}]`.
The pitch is x ∈ [-55, 55], y ∈ [-35, 35]; home defends −x. Every answer is scored for latency and validity,
and a slow or invalid answer means the player stands still for that tick.

### 2. Request pipeline (`lib/dcon_core.py → build_handler`)
1. **Parse & identify** – read `teamId` / `myPlayers`; accept numeric, `"home"/"away"` or `agentId_N` forms.
2. **`normalize_state()`** – the server format changed during the event (`teamCode`, `agentId`,
   `possessionAgentId` without a team). We map every player to `home/away`, every id to `agentId_N`,
   and positions to `{x, y}`. When team labels are unknown (e.g. UUIDs) the side with the lower mean x is
   *home*. This single function fixed the biggest bug we had: the agents never recognised that *they*
   had the ball, so every pass/shot was being converted into a run.
3. **`tactics()` – deterministic layer (~0 ms).** Covers *every* situation:
   * **On the ball** – shoot if inside the role's range (FWD 24 m, 32 m with an open lane; MID 20/26 m;
     DEF 20/24 m) aiming at the corner the keeper is *not* covering (`aim_away_from_gk`), power 0.8–0.9 for
     accuracy; otherwise pass to the best target (`best_pass_target` scores progress, free space and a clear
     passing lane via point-to-segment distance); pass immediately under pressure (< 5 m); else dribble forward.
   * **Goalkeeper** – stands on the ball→goal-centre line, 1.5–2.5 m off the line, |y| ≤ 3.5 m
     (angle narrowing). Comes out (`INTERCEPT`) only for a loose ball inside 9 m in the box and closes down
     (`PRESS_BALL`) a shooter inside 12 m. Distributes with `GK_DISTRIBUTE` (throw < 25 m, kick beyond).
   * **Defending** – exactly **one** presser (the closest outfield player); the defender blocks the
     ball→goal line at 40 % and slide-tackles inside 3.5 m; the midfielder screens at 65 % or cuts passing
     lanes; forwards hold a counter-attack position.
   * **Attacking shape** – forwards run ahead of the ball into the box (y = ±7), the midfielder supports
     8 m behind and offset, the defender holds a high line.
   * **No data yet (pre-kick-off)** – go to the formation spot instead of waiting for a model.
4. **LLM layer (optional)** – `ModelChain` runs a Strands `Agent` with a role prompt in a thread pool with
   a hard timeout. After 3 slow answers it degrades to the next model (Sonnet → Nova Micro) and retries the
   primary after 25 good answers. In v8 it is never reached during a match because `tactics()` covers every
   case.
5. **Safety net** – `sanitize()` swaps ball-only commands issued without the ball, drops wasted commands
   (`SET_STANCE`, `CLEAR_OVERRIDE`); `enforce_allowed()` maps a command that isn't unlocked to the closest
   unlocked one; any exception falls back to the sample's rule-based fallback, and a last-resort `MOVE_TO`.
   The agent never returns nothing.

### 3. Why we stopped asking an LLM every tick
| Model (Bedrock Converse) | Measured | Problem |
|---|---|---|
| Claude Opus 4.6 (global) | ~2.2 s | platform marked our players idle |
| Claude Sonnet 4.6 (global) | ~1.7 s | still > 1 tick |
| Amazon Nova Micro | 0.3–0.9 s | fast, but weak decisions and some invalid JSON |
| Rule layer | ~0 ms compute, ~100 ms round-trip | deterministic, always valid |

A 2-second tick means a 2-second decision is already stale: the ball has moved. The geometry of football
(distance to goal, lanes, who is closest) is cheap to compute exactly, so we moved it into code and kept
the LLM for what it is good at — reading free-text coach instructions — as an optional layer.

### 4. Debugging with data
* **Match reports** (command mix, per-agent latency, shots) told us *what* went wrong: 27 % `PRESS_BALL`
  vs 7 % for the house team, 0 shots on target, 600–870 ms latency.
* **CloudWatch logs** (`/aws/bedrock-agentcore/runtimes/<agent>-DEFAULT`, one stream per match) told us
  *why*: we log a `PAYLOAD`/`RAW` sample every 15 calls and every `RULE` / `LLM` decision with its ms.
  That is how we found the "never has the ball" bug, the single slow LLM call at kick-off, and that our own
  compute was 0 ms (the rest is network + runtime).
* **Offline tests** (`test_dcon.py`) mock Strands/Bedrock and assert: right player/team ids, slow model
  still answers in < 0.7 s, errors fall back, forwards shoot in range, GK distributes, away team works.
  A random-state simulation checks the command mix before every deploy.

### 5. Deployment
`agentcore deploy --auto-update-on-conflict` per position, run **in parallel** for all five agents
(≈ 3.5 min instead of ≈ 15 min), keeping the same runtime names so the ARNs registered in the portal never
change. Code was moved into AWS CloudShell as gzip+base64 chunks and verified with `md5sum` before each
deploy. A pre-match fitness drill (5 ticks per agent) is run after every deploy.

## Layout

```
lib/dcon_core.py      core: normalisation, tactics, model chain, handler
lib/dcon_prompts.py   system prompts per role (hybrid of the workshop prompt library)
lib/dcon_models.json  model chain + timeout per position
lib/*.py              AWS sample helpers (state, parsing, fallback)
agents/ai-*/src/main.py  one AgentCore runtime per position
test_dcon.py          offline tests (mocked Bedrock)
bench_dcon.py         real Bedrock latency / JSON benchmark
```

## Run

```bash
python3 test_dcon.py                 # offline tests
cd agents && ./deploy-all.sh ai-gk   # deploy one position with the AgentCore starter toolkit
```

Deploying needs your own AWS account with Bedrock AgentCore access. No credentials are stored in this repo.

## Team
Coach: Mirkazim Kazimov — [LinkedIn](https://linkedin.com/in/mirkazimkazimov) · [GitHub](https://github.com/kazimovmirkazim)
