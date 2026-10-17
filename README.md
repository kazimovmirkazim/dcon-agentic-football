# DCON — AWS Agentic Football Cup (Baku, 2026)

Five autonomous football agents (GK, DEF, MID, 2×FWD) built for the **AWS Agentic Football Cup**, deployed on
**Amazon Bedrock AgentCore Runtime**. Team **DCON** — players *Titan* (GK), *Reaper* (DEF), *Apex* (MID),
*Ares* and *Zero* (FWD).

Built on top of the official AWS sample:
[aws-samples/sample-ai-possibilities → agentic-football-sample-agents](https://github.com/aws-samples/sample-ai-possibilities)
(MIT-0, see `LICENSE-aws-sample`).

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
