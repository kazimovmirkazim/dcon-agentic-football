"""DCON local test - no AWS needed. Run from agentic-football-sample-agents/:  python3 test_dcon.py"""
import sys, json, time, types, asyncio, copy
sys.path.insert(0, 'lib')
from test_helpers import mock_agentcore, GAME_STATE
import test_helpers
mock_agentcore()
test_helpers._FakeApp.entrypoint = lambda self, fn: setattr(self, "_fn", fn) or fn

MODE = {"m": "ok"}


class FakeModel:
    def __init__(self, model_id, **kw):
        self.model_id = model_id


class FakeAgent:
    calls = 0
    last = ""

    def __init__(self, model, system_prompt, callback_handler=None):
        self.m = model

    def __call__(self, text):
        FakeAgent.calls += 1
        FakeAgent.last = text
        if MODE["m"] == "slow":
            time.sleep(1.0)
        if MODE["m"] == "err":
            raise RuntimeError("AccessDenied")
        return '[{"commandType":"PASS","playerId":9,"parameters":{"target_player_id":4,"type":"THROUGH"},"duration":0}]'


s = types.ModuleType("strands"); s.Agent = FakeAgent; sys.modules["strands"] = s
sm = types.ModuleType("strands.models"); sm.BedrockModel = FakeModel; sys.modules["strands.models"] = sm

import dcon_core
dcon_core.load_model_chain = lambda label: (["fake-a", "fake-b"], 0.5)

import importlib.util


def give(gs, idx):
    """Give the ball to HOME player idx (ball placed at the player)."""
    gs["ball"]["possessionAgentId"] = f"agentId_{idx}"
    p = next(p for p in gs["players"] if p["agentId"] == f"agentId_{idx}" and p["teamCode"] == "home")
    gs["ball"]["position"] = dict(p["position"], z=0)


def load(a):
    spec = importlib.util.spec_from_file_location(f"m_{a}_{time.time()}", f"ai-team-dcon/ai-{a}/src/main.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def run(mod, gs, team=0):
    fn = mod.app._fn

    async def go():
        out = []
        async for x in fn({"prompt": json.dumps({"gameState": gs, "teamId": team, "myPlayers": [mod.MY_PLAYER_ID]})}, None):
            out.append(x)
        return json.loads(out[0])
    return asyncio.run(go())


ok = True
for a in ["gk", "def", "mid", "fwd1", "fwd2"]:
    m = load(a)
    gs = copy.deepcopy(GAME_STATE); give(gs, 1 if a == "mid" else 2)
    MODE["m"] = "ok"; r = run(m, gs)
    ok &= r[0]["playerId"] == m.MY_PLAYER_ID and r[0]["teamId"] == 0
    print(f"{a:5s} normal -> {r[0]['commandType']:14s}", end=" | ")
    MODE["m"] = "slow"; t = time.time(); r = run(m, gs); dt = time.time() - t
    print(f"slow model -> {r[0]['commandType']:9s} in {dt*1000:4.0f}ms", end=" | "); ok &= dt < 0.7
    MODE["m"] = "err"; r = run(m, gs); print(f"error -> {r[0]['commandType']}")
    ok &= r[0]["playerId"] == m.MY_PLAYER_ID

m = load("fwd1"); gs = copy.deepcopy(GAME_STATE)
gs["players"][3]["position"] = {"x": 40, "y": -3}; give(gs, 3)
n = FakeAgent.calls; MODE["m"] = "ok"; r = run(m, gs)
print("FWD reflex  ->", r[0]["commandType"], r[0]["parameters"], "| LLM calls:", FakeAgent.calls - n)
ok &= r[0]["commandType"] == "SHOOT" and FakeAgent.calls == n

m = load("gk"); gs = copy.deepcopy(GAME_STATE); give(gs, 0); r = run(m, gs)
print("GK reflex   ->", r[0]["commandType"], r[0]["parameters"]); ok &= r[0]["commandType"] == "GK_DISTRIBUTE"

m = load("fwd2"); r = run(m, copy.deepcopy(GAME_STATE), team=1); ok &= r[0]["teamId"] == 1
print("AWAY team   ->", r[0]["commandType"], "teamId", r[0]["teamId"])

m = load("fwd2"); gs = copy.deepcopy(GAME_STATE); give(gs, 1)
gs["teamChat"] = ["Zero, stay high and shoot early!", {"teamCode": "away", "message": "Rincon press"}]
run(m, gs)
print("--- what Zero's LLM sees (tail) ---"); print("\n".join(FakeAgent.last.splitlines()[-11:]))
ok &= "Zero, stay high" in FakeAgent.last and "[OTHER TEAM" in FakeAgent.last
# opponent GK holds the ball -> our GK must NOT distribute
m = load("gk"); gs = copy.deepcopy(GAME_STATE); gs["ball"]["possessionAgentId"] = "agentId_0"; gs["ball"]["position"] = {"x": 49.5, "y": 0, "z": 0}
r = run(m, gs); print("opp GK has ball -> our GK:", r[0]["commandType"]); ok &= r[0]["commandType"] != "GK_DISTRIBUTE"
print("\nALL OK" if ok else "\nSOMETHING FAILED")
sys.exit(0 if ok else 1)
