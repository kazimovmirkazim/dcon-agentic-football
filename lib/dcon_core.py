"""DCON core: richer game-state summary, coach (teamChat) handling, reflex layer,
time-boxed LLM calls with a model chain, and never-freeze fallbacks.

Built on top of the AWS sample lib (state.py, parsing.py, fallback.py).
"""
import concurrent.futures as cf
import json
import math
import os
import time

from state import (summarize_state, get_goal_positions, _player_idx, _is_my_team,
                   _possession_idx, dist)
from parsing import parse_commands
from fallback import build_last_resort

NAMES = {0: "Titan", 1: "Reaper", 2: "Apex", 3: "Ares", 4: "Zero"}
ROLE_OF = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}
TEAM_NAME = "DCON"

_POOL = cf.ThreadPoolExecutor(max_workers=4)
_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CHAIN = ["us.amazon.nova-micro-v1:0"]


def load_model_chain(label: str) -> list:
    """Model chain for a position from dcon_models.json (best first, fastest last)."""
    try:
        with open(os.path.join(_HERE, "dcon_models.json")) as f:
            cfg = json.load(f)
        chain = cfg.get(label) or cfg.get("default") or DEFAULT_CHAIN
        timeout = float(cfg.get("timeout_s", 1.3))
        return list(chain), timeout
    except Exception:
        return list(DEFAULT_CHAIN), 1.3


# ---------------------------------------------------------------------------
# Coach messages (teamChat)
# ---------------------------------------------------------------------------
_coach = {"seen": [], "last_time": -1.0}


def _msg_text(m):
    if isinstance(m, str):
        return m.strip(), None
    if isinstance(m, dict):
        txt = ""
        for k in ("message", "text", "content", "msg", "instruction"):
            if isinstance(m.get(k), str) and m[k].strip():
                txt = m[k].strip()
                break
        sender = None
        for k in ("teamCode", "team", "sender", "from", "teamId"):
            if m.get(k) is not None:
                sender = m.get(k)
                break
        return txt, sender
    return str(m), None


def coach_lines(game_state: dict, team_id: int, max_n: int = 4) -> list:
    """Keep a per-match history of coach messages; return the latest few."""
    t = float(game_state.get("gameTime", 0) or 0)
    if t + 5 < _coach["last_time"]:          # new match started -> reset history
        _coach["seen"] = []
    _coach["last_time"] = t
    chat = game_state.get("teamChat") or []
    if isinstance(chat, (str, dict)):
        chat = [chat]
    my_code = "home" if team_id == 0 else "away"
    for m in chat:
        txt, sender = _msg_text(m)
        if not txt:
            continue
        if sender is not None and str(sender) not in (my_code, str(team_id), TEAM_NAME):
            # explicitly from the other side -> keep but mark
            txt = f"[OTHER TEAM - ignore] {txt}"
        if txt not in _coach["seen"]:
            _coach["seen"].append(txt)
    return _coach["seen"][-max_n:]


# ---------------------------------------------------------------------------
# State normalisation: make team / index / position fields uniform whatever the server sends
# ---------------------------------------------------------------------------
_HOME = ("home", "0", "h", "team0", "left", "team_0", "a")
_AWAY = ("away", "1", "team1", "right", "team_1", "b")


def _raw_team(p):
    for k in ("teamCode", "team", "side", "teamId", "teamIndex", "teamSide"):
        v = p.get(k)
        if v is not None:
            return str(v).strip().lower()
    return None


def normalize_state(gs):
    players = gs.get("players") or []
    if isinstance(players, dict):
        players = list(players.values())
        gs["players"] = players
    for p in players:
        if "position" not in p and "x" in p:
            p["position"] = {"x": p.get("x", 0), "y": p.get("y", 0)}
        if "agentId" not in p:
            for k in ("playerId", "id", "index", "playerIndex"):
                if p.get(k) is not None:
                    p["agentId"] = f"agentId_{str(p[k]).rsplit('_', 1)[-1]}"
                    break
    vals = {_raw_team(p) for p in players}
    unknown = [v for v in vals if v not in _HOME and v not in _AWAY]
    mapping = {}
    if unknown:
        # unknown team labels (e.g. team UUIDs): home = side whose players sit at lower x on average
        avg = {}
        for v in vals:
            xs = [p.get("position", {}).get("x", 0) for p in players if _raw_team(p) == v]
            avg[v] = sum(xs) / max(1, len(xs))
        order = sorted(avg, key=avg.get)
        if len(order) >= 2:
            mapping = {order[0]: "home", order[-1]: "away"}
    for p in players:
        v = _raw_team(p)
        code = "home" if v in _HOME else "away" if v in _AWAY else mapping.get(v)
        if code:
            p["teamCode"] = code
    ball = gs.get("ball") or {}
    if "position" not in ball and "x" in ball:
        ball["position"] = {"x": ball.get("x", 0), "y": ball.get("y", 0)}
    pa = ball.get("possessionAgentId")
    if pa is None and ball.get("possessionPlayerId") is not None:
        pa = ball["possessionAgentId"] = f"agentId_{ball['possessionPlayerId']}"
    if isinstance(pa, str) and not ball.get("possessionTeamCode"):
        low = pa.lower()
        if "home" in low or "team0" in low:
            ball["possessionTeamCode"] = "home"
        elif "away" in low or "team1" in low:
            ball["possessionTeamCode"] = "away"
    tc = ball.get("possessionTeamCode") or ball.get("possessionTeam") or ball.get("possessionTeamId")
    if tc is not None:
        t = str(tc).lower()
        ball["possessionTeamCode"] = "home" if t in _HOME else "away" if t in _AWAY else mapping.get(t, tc)
    gs["ball"] = ball
    return gs


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _split(players, team_id):
    mine = [p for p in players if _is_my_team(p, team_id)]
    opp = [p for p in players if not _is_my_team(p, team_id)]
    return mine, opp


def _seg_dist(p, a, b):
    ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
    px, py = p.get("x", 0), p.get("y", 0)
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    if L == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def lane_open(frm, to, opponents, width=3.0, skip_gk=False):
    for o in opponents:
        if skip_gk and _player_idx(o) == 0:
            continue
        if _seg_dist(o.get("position", {}), frm, to) < width:
            return False
    return True


def aim_away_from_gk(opponents, opp_goal_x):
    """Low shot to the side of the goal the keeper is NOT covering."""
    gk = next((o for o in opponents if _player_idx(o) == 0), None)
    gy = gk.get("position", {}).get("y", 0) if gk else 0
    left_sign = 1 if opp_goal_x > 0 else -1          # attacker's left is +y when attacking +x
    return "BR" if gy * left_sign > 0 else "BL"


def ball_holder(game_state):
    """(player, team_code) of the ball holder. possessionAgentId has no team, so use an explicit
    team field if the server sends one, else the player with that index closest to the ball."""
    ball = game_state.get("ball", {}) or {}
    idx = _possession_idx(ball)
    if idx is None:
        return None
    bpos = ball.get("position", {"x": 0, "y": 0})
    cands = [p for p in game_state.get("players", []) if _player_idx(p) == idx]
    tcode = ball.get("possessionTeamCode") or ball.get("possessionTeam")
    if tcode is not None:
        for p in cands:
            if p.get("teamCode") == tcode or str(p.get("teamId")) == str(tcode):
                return p
    return min(cands, key=lambda p: dist(p.get("position", {}), bpos), default=None)


def i_have_ball(game_state, team_id, my_id):
    h = ball_holder(game_state)
    return bool(h) and _player_idx(h) == my_id and _is_my_team(h, team_id)


def best_pass_target(me_id, me_pos, mine, opponents, opp_goal_x, exclude=(0,)):
    best, best_score = None, -1e9
    for p in mine:
        pid = _player_idx(p)
        if pid == me_id or pid in exclude:
            continue
        pos = p.get("position", {})
        free = min((dist(pos, o.get("position", {})) for o in opponents), default=99)
        lane = lane_open(me_pos, pos, opponents, width=2.5)
        advance = -abs(pos.get("x", 0) - opp_goal_x)
        score = advance * 0.6 + min(free, 15) * 1.5 + (10 if lane else -25)
        if score > best_score:
            best, best_score = p, score
    return best


# ---------------------------------------------------------------------------
# Rich summary
# ---------------------------------------------------------------------------
def rich_summary(game_state, team_id, my_id, label):
    base = summarize_state(game_state, team_id, my_id, label)
    players = game_state.get("players", [])
    ball = game_state.get("ball", {})
    bpos = ball.get("position", {"x": 0, "y": 0})
    mine, opp = _split(players, team_id)
    me = next((p for p in mine if _player_idx(p) == my_id), None)
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    holder = ball_holder(game_state)
    has_ball = i_have_ball(game_state, team_id, my_id)
    # correct the sample summary: team-aware ball holder, stamina in %
    import re as _re
    if holder is not None:
        side = "MY" if _is_my_team(holder, team_id) else "OPP"
        base = _re.sub(r"held by [^\n]*", f"held by {side} player {_player_idx(holder)}", base, count=1)
    base = _re.sub(r"hasBall=\w+", f"hasBall={has_ball}", base)
    if me:
        st_ = me.get("stamina", 1.0)
        base = _re.sub(r"stam=\S+", f"stam={st_*100 if st_ <= 1 else st_:.0f}%", base, count=1)
    lines = [base, "", f"=== DCON TACTICAL NOTES (you are {NAMES.get(my_id, my_id)}, {label}) ==="]
    if me:
        pos = me.get("position", {})
        stam = me.get("stamina", 1.0)
        stam_pct = stam * 100 if stam <= 1.0 else stam
        lines.append(f"Your stamina: {stam_pct:.0f}% ({'LOW - do not sprint' if stam_pct < 40 else 'ok to sprint'})")
        near = min(((dist(pos, o.get('position', {})), _player_idx(o)) for o in opp), default=(99, -1))
        lines.append(f"Nearest opponent: P{near[1]} at {near[0]:.1f}")
        goal = {"x": opp_goal_x, "y": 0}
        d_goal = abs(pos.get("x", 0) - opp_goal_x)
        lines.append(f"Distance to opponent goal: {d_goal:.1f} | shot lane: "
                     f"{'OPEN' if lane_open(pos, goal, opp, 2.5, skip_gk=True) else 'BLOCKED'} | "
                     f"suggested aim: {aim_away_from_gk(opp, opp_goal_x)}")
        lines.append(f"You have the ball: {'YES' if has_ball else 'no'}")
        bdist = {_player_idx(p): dist(p.get("position", {}), bpos) for p in mine}
        closest = min(bdist, key=bdist.get) if bdist else None
        lines.append(f"Closest DCON player to the ball: {'YOU - you may press/chase' if closest == my_id else NAMES.get(closest, closest) + ' - do NOT press, keep your position/space'}")
        tgt = best_pass_target(my_id, pos, mine, opp, opp_goal_x,
                               exclude=(0,) if my_id != 0 else ()) if has_ball else None
        if tgt:
            tid = _player_idx(tgt)
            lines.append(f"Best pass option: id={tid} ({NAMES.get(tid, tid)}), "
                         f"distOppGoal={abs(tgt.get('position', {}).get('x', 0) - opp_goal_x):.1f}")
    s = game_state.get("score", {})
    ours = s.get("home", 0) if team_id == 0 else s.get("away", 0)
    theirs = s.get("away", 0) if team_id == 0 else s.get("home", 0)
    state = "WINNING" if ours > theirs else "LOSING" if ours < theirs else "DRAWING"
    t = float(game_state.get("gameTime", 0) or 0)
    pm = str(game_state.get("playMode", ""))
    if "KICK" in pm.upper():
        lines.append("KICK-OFF: take your formation spot fast; forwards start the attack immediately")
    lines.append(f"Match: {state} {ours}-{theirs} at {t:.0f}s"
                 + (" | GOLDEN GOAL RISK: next goal may decide the match" if state == "DRAWING" and t > 100 else ""))
    if holder is not None:
        lines.append("Ball: " + ("WE have it -> attack" if _is_my_team(holder, team_id) else "THEY have it -> win it back"))
    else:
        lines.append("Ball: LOOSE -> nearest player go for it")
    coach = coach_lines(game_state, team_id)
    if coach:
        lines.append("COACH MESSAGES (latest last):")
        lines += [f"  - {c}" for c in coach]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reflex layer: instant decisions that do not need an LLM
# ---------------------------------------------------------------------------
def _cmd(t, pid, tid, params, duration=0):
    return {"commandType": t, "playerId": pid, "teamId": tid, "parameters": params, "duration": duration}


def reflex(game_state, team_id, my_id, shoot_now):
    players = game_state.get("players", [])
    ball = game_state.get("ball", {})
    if not i_have_ball(game_state, team_id, my_id):
        return None
    mine, opp = _split(players, team_id)
    me = next((p for p in mine if _player_idx(p) == my_id), None)
    if not me:
        return None
    pos = me.get("position", {})
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    if my_id == 0:  # goalkeeper: distribute immediately
        tgt = best_pass_target(0, pos, mine, opp, opp_goal_x, exclude=())
        if tgt:
            far = dist(pos, tgt.get("position", {})) > 25
            return [_cmd("GK_DISTRIBUTE", 0, team_id, {"target_player_id": _player_idx(tgt),
                                                         "method": "KICK" if far else "THROW"})]
        return None
    d_goal = abs(pos.get("x", 0) - opp_goal_x)
    if shoot_now and d_goal <= shoot_now:
        return [_cmd("SHOOT", my_id, team_id, {"aim_location": aim_away_from_gk(opp, opp_goal_x), "power": 0.95})]
    return None


def tactics(game_state, team_id, my_id, shoot_now):
    """Deterministic layer for clear-cut moments (defence + ball actions). None -> ask the LLM."""
    players = game_state.get("players", [])
    ball = game_state.get("ball", {}) or {}
    bpos = ball.get("position", {"x": 0, "y": 0})
    mine, opp = _split(players, team_id)
    me = next((p for p in mine if _player_idx(p) == my_id), None)
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    if not me:   # no player data yet (pre-kick-off): go to formation spot instantly, never wait for the LLM
        sgn = 1 if my_goal_x < 0 else -1
        fx, fy = {0: (-50, 0), 1: (-30, 0), 2: (-12, 0), 3: (-3, -8), 4: (-3, 8)}.get(my_id, (0, 0))
        return [_cmd("MOVE_TO", my_id, team_id, {"target_x": fx * sgn, "target_y": fy, "sprint": False})]
    pos = me.get("position", {"x": 0, "y": 0})
    holder = ball_holder(game_state)
    we_have = holder is not None and _is_my_team(holder, team_id)
    they_have = holder is not None and not we_have
    i_have = we_have and _player_idx(holder) == my_id
    d_ball = dist(pos, bpos)
    field = [p for p in mine if _player_idx(p) != 0]
    closest = min(field, key=lambda p: dist(p.get("position", {}), bpos)) if field else None
    i_am_closest = closest is not None and _player_idx(closest) == my_id
    ball_in_our_half = (bpos.get("x", 0) - 0) * (1 if my_goal_x < 0 else -1) < 0
    goal = {"x": opp_goal_x, "y": 0}

    # --- with the ball: always decide instantly (pass / shoot / dribble), never wait for the LLM ---
    if i_have:
        if my_id == 0:
            return reflex(game_state, team_id, my_id, 0) or [
                _cmd("GK_DISTRIBUTE", 0, team_id, {"target_player_id": 2, "method": "KICK"})]
        d_goal = abs(pos.get("x", 0) - opp_goal_x)
        open_lane = lane_open(pos, goal, opp, 2.5, skip_gk=True)
        aim = aim_away_from_gk(opp, opp_goal_x)
        pw = 0.8 if d_goal < 18 else 0.9
        if my_id in (2, 3, 4) and (d_goal <= (24 if my_id in (3, 4) else 20)
                                   or (open_lane and d_goal <= (32 if my_id in (3, 4) else 26))):
            return [_cmd("SHOOT", my_id, team_id, {"aim_location": aim, "power": pw})]
        if my_id == 1 and (d_goal <= 20 or (open_lane and d_goal <= 24)):
            return [_cmd("SHOOT", my_id, team_id, {"aim_location": aim, "power": pw})]
        pressure = min((dist(pos, o.get("position", {})) for o in opp), default=99)
        tgt = best_pass_target(my_id, pos, mine, opp, opp_goal_x, exclude=(0, my_id))
        t_goal = abs(tgt.get("position", {}).get("x", 0) - opp_goal_x) if tgt else 99
        t_lane = tgt is not None and lane_open(pos, tgt.get("position", {}), opp, 2.0)
        # pass when a team-mate is further forward with a clear lane, or when we are under pressure
        if tgt and t_lane and (t_goal < d_goal - 4 or pressure < 5 or my_id == 1):
            ahead = t_goal < 30
            return [_cmd("PASS", my_id, team_id, {"target_player_id": _player_idx(tgt),
                                                   "type": "THROUGH" if ahead else "GROUND"})]
        if tgt and pressure < 3:
            return [_cmd("PASS", my_id, team_id, {"target_player_id": _player_idx(tgt), "type": "AERIAL"})]
        # dribble towards goal, drifting to the side away from the nearest defender
        step = 8 if opp_goal_x > 0 else -8
        ty = pos.get("y", 0) * 0.6
        return [_cmd("MOVE_TO", my_id, team_id, {"target_x": pos.get("x", 0) + step, "target_y": ty, "sprint": True})]

    # --- goalkeeper: shot-stopper, narrows the angle, rushes out only for clear chances ---
    if my_id == 0:
        bx_from_goal = abs(bpos.get("x", 0) - my_goal_x)
        in_box = bx_from_goal < 16 and abs(bpos.get("y", 0)) < 20
        if not we_have and in_box and d_ball < 9:
            return [_cmd("INTERCEPT", 0, team_id, {"aggressive": True})]   # loose ball / 1v1 in the box
        if they_have and in_box and d_ball < 12:
            return [_cmd("PRESS_BALL", 0, team_id, {"intensity": 1.0}, duration=1)]  # close down the shooter
        # stand on the line ball -> goal centre: the further the ball, the closer to the line
        depth = 2.5 if bx_from_goal < 30 else 1.5
        ratio = depth / max(bx_from_goal, 1.0)
        gx = my_goal_x + (depth if my_goal_x < 0 else -depth)
        gy = max(-3.5, min(3.5, bpos.get("y", 0) * ratio * 1.6))
        return [_cmd("MOVE_TO", 0, team_id, {"target_x": gx, "target_y": gy, "sprint": True})]

    # --- every remaining case is rule-based too: no LLM wait, GK-like latency ---
    stam = me.get("stamina", 1.0)
    stam = stam * 100 if stam <= 1.0 else stam
    run = stam > 25
    fwd = 1 if opp_goal_x > 0 else -1          # +1 when we attack towards +x
    bx, by = bpos.get("x", 0), bpos.get("y", 0)
    ball_in_their_half = not ball_in_our_half

    def mv(x, y, sprint=run):
        x = max(-52.0, min(52.0, x)); y = max(-32.0, min(32.0, y))
        return [_cmd("MOVE_TO", my_id, team_id, {"target_x": x, "target_y": y, "sprint": bool(sprint)})]

    if they_have:
        h = holder.get("position", {})
        if dist(pos, h) < 3.5 and (my_id == 1 or ball_in_our_half):
            return [_cmd("SLIDE_TACKLE", my_id, team_id, {"target_player_id": _player_idx(holder), "sprint": True, "distance": 4.0})]
        if i_am_closest and (d_ball < 14 or ball_in_our_half):
            return [_cmd("PRESS_BALL", my_id, team_id, {"intensity": 1.0}, duration=2)]
        if i_am_closest:
            return mv(bx, by, True)
        if my_id == 1:   # Reaper: block the line ball -> goal
            return mv(my_goal_x + (bx - my_goal_x) * 0.4, by * 0.4, ball_in_our_half)
        if my_id == 2:   # Apex: second presser high up, screen when deep
            if ball_in_their_half:
                return mv(bx - 10 * fwd, by * 0.5, False)   # cut the passing lane
            return mv(my_goal_x + (bx - my_goal_x) * 0.65, by * 0.6, True)
        # forwards: aggressive high press in their half, otherwise wait for the counter
        return mv(fwd * 6, -9 if my_id == 3 else 9, False)

    if holder is None:   # loose ball: nearest wins it, the rest take attacking shape
        if i_am_closest:
            if d_ball < 10:
                return [_cmd("INTERCEPT", my_id, team_id, {"aggressive": True})]
            return mv(bx, by, True)

    # we have the ball (or it is loose): attacking shape, forwards attack the box
    if my_id in (3, 4):
        side = -7 if my_id == 3 else 7
        tx = max(bx * fwd + 10, 30) * fwd           # stay ahead of the ball, at least 30 up the pitch
        tx = min(abs(tx), 47) * fwd
        return mv(tx, side)
    if my_id == 2:       # Apex: support 8 behind the ball, offset to open a lane
        return mv(bx - 8 * fwd, by + (8 if by <= 0 else -8))
    if my_id == 1:       # Reaper: hold a high line behind the ball
        return mv(my_goal_x + (bx - my_goal_x) * 0.45, by * 0.3, False)
    return None


# ---------------------------------------------------------------------------
# Time-boxed LLM with model chain
# ---------------------------------------------------------------------------
class ModelChain:
    def __init__(self, system_prompt, models, max_tokens=90, log=None):
        self.sp = system_prompt
        self.models = models
        self.i = 0
        self.max_tokens = max_tokens
        self.slow = 0
        self.ok_since_degrade = 0
        self.log = log
        self._models = {}

    @property
    def current(self):
        return self.models[self.i]

    def _model(self, mid):
        if mid not in self._models:
            from strands.models import BedrockModel
            self._models[mid] = BedrockModel(model_id=mid, max_tokens=self.max_tokens, temperature=0.2)
        return self._models[mid]

    def _run(self, mid, text):
        from strands import Agent
        a = Agent(model=self._model(mid), system_prompt=self.sp, callback_handler=None)  # fresh context each tick
        return str(a(text))

    def _degrade(self, why):
        if self.i < len(self.models) - 1:
            self.i += 1
            if self.log:
                self.log.warn(f"switching model ({why}) -> {self.current}")

    def call(self, text, timeout):
        fut = _POOL.submit(self._run, self.current, text)
        try:
            out = fut.result(timeout=timeout)
            self.slow = max(0, self.slow - 1)
            if self.i > 0:                      # periodically retry the primary (e.g. after throttling)
                self.ok_since_degrade += 1
                if self.ok_since_degrade >= 25:
                    self.ok_since_degrade = 0
                    self.i = 0
                    if self.log:
                        self.log.info(f"retrying primary model {self.current}")
            return out
        except cf.TimeoutError:
            self.slow += 1
            if self.slow >= 3:
                self.slow = 0
                self._degrade("too slow")
            raise
        except Exception as e:
            self._degrade(f"error {type(e).__name__}")
            raise


BALL_ONLY = {"SHOOT", "PASS", "GK_DISTRIBUTE"}
USELESS = {"SET_STANCE", "CLEAR_OVERRIDE", "RESET"}


def sanitize(cmds, game_state, team_id, my_id, fallback_fn):
    """Ball-only commands without the ball waste a tick: replace them."""
    if cmds and cmds[0].get("commandType") in USELESS:
        return fallback_fn(game_state, team_id, my_id)
    if not cmds or cmds[0].get("commandType") not in BALL_ONLY:
        return cmds
    if i_have_ball(game_state, team_id, my_id):
        return cmds
    _, opp_goal_x = get_goal_positions(team_id)
    if my_id in (3, 4):  # forwards: run into the box instead
        y = -8 if my_id == 3 else 6
        return [_cmd("MOVE_TO", my_id, team_id, {"target_x": opp_goal_x * 0.75, "target_y": y, "sprint": True})]
    return fallback_fn(game_state, team_id, my_id)


def team_safe(fallback_fn):
    """The sample fallback reads possessionAgentId without the team. If the opponent holds
    the ball, hide possession so the fallback never issues ball-only commands for us."""
    def wrapped(game_state, team_id, my_id):
        gs = game_state
        h = ball_holder(game_state)
        if h is not None and not _is_my_team(h, team_id):
            gs = dict(game_state)
            gs["ball"] = dict(game_state.get("ball", {}))
            gs["ball"]["possessionAgentId"] = None
            gs["ball"]["possessionPlayerId"] = None
        return fallback_fn(gs, team_id, my_id)
    return wrapped


_stats = {"n": 0}


def enforce_allowed(cmds, allowed, game_state, team_id, my_id):
    """Map any command the server has not unlocked for us to an allowed equivalent."""
    if not allowed or not cmds:
        return cmds
    out = []
    ball = (game_state.get("ball", {}) or {}).get("position", {"x": 0, "y": 0})
    _, opp_goal_x = get_goal_positions(team_id)
    me = next((p for p in game_state.get("players", []) if _player_idx(p) == my_id and _is_my_team(p, team_id)), None)
    my_pos = me.get("position", {"x": 0, "y": 0}) if me else {"x": 0, "y": 0}
    for c in cmds:
        t = c.get("commandType")
        if t in allowed:
            out.append(c); continue
        alt = None
        if t in ("PRESS_BALL", "INTERCEPT", "SLIDE_TACKLE", "MARK", "FOLLOW_PLAYER"):
            if "PRESS_BALL" in allowed and t != "PRESS_BALL":
                alt = _cmd("PRESS_BALL", my_id, team_id, {"intensity": 0.8}, duration=2)
            elif "INTERCEPT" in allowed and t != "INTERCEPT":
                alt = _cmd("INTERCEPT", my_id, team_id, {"aggressive": True})
            elif "MOVE_TO" in allowed:
                alt = _cmd("MOVE_TO", my_id, team_id, {"target_x": ball.get("x", 0), "target_y": ball.get("y", 0), "sprint": True})
        elif t == "GK_DISTRIBUTE" and "PASS" in allowed:
            alt = _cmd("PASS", my_id, team_id, {"target_player_id": c.get("parameters", {}).get("target_player_id", 1), "type": "GROUND"})
        elif t == "SHOOT" and "MOVE_TO" in allowed:
            alt = _cmd("MOVE_TO", my_id, team_id, {"target_x": opp_goal_x, "target_y": 0, "sprint": True})
        elif t == "PASS" and "MOVE_TO" in allowed:
            alt = _cmd("MOVE_TO", my_id, team_id, {"target_x": opp_goal_x * 0.6, "target_y": my_pos.get("y", 0), "sprint": False})
        if alt is None and "MOVE_TO" in allowed:
            alt = _cmd("MOVE_TO", my_id, team_id, {"target_x": my_pos.get("x", 0), "target_y": my_pos.get("y", 0), "sprint": False})
        if alt is not None:
            out.append(alt)
    return out or cmds


def build_handler(app, label, my_id, system_prompt, fallback_fn, fallback_cfg, shoot_now=0):
    log = app.logger
    fallback_fn = team_safe(fallback_fn)
    models, timeout = load_model_chain(label)
    chain = ModelChain(system_prompt, models, log=log)
    last_resort = build_last_resort(fallback_cfg, my_id)

    @app.entrypoint
    async def invoke(payload, context):
        t0 = time.time()
        team_id, pid, gs, allowed = 0, my_id, {}, None
        try:
            prompt = payload.get("prompt", "{}")
            data = json.loads(prompt) if isinstance(prompt, str) else prompt
            gs = data.get("gameState", {}) or {}
            if _stats["n"] % 15 == 0:
                try:
                    log.info(f"{label} RAW p0={json.dumps((gs.get('players') or [None])[0])[:300]} "
                             f"ball={json.dumps(gs.get('ball'))[:300]} modeTeamId={gs.get('modeTeamId')!r}")
                except Exception:
                    pass
            gs = normalize_state(gs)
            team_id = data.get("teamId", 0)
            # some servers send the side as text; trust it over a numeric default
            for k in ("teamCode", "side", "team", "homeAway"):
                v = data.get(k)
                if isinstance(v, str) and v.lower() in ("home", "away"):
                    team_id = 0 if v.lower() == "home" else 1
            if isinstance(team_id, str):
                team_id = 1 if team_id.lower() in ("1", "away") else 0
            mp = data.get("myPlayers", [my_id])
            pid = mp[0] if mp else my_id
            if isinstance(pid, str):
                try:
                    pid = int(pid.rsplit("_", 1)[-1])
                except ValueError:
                    pid = my_id
            allowed = data.get("unlockedCommands")
            allowed = set(allowed) if isinstance(allowed, (list, tuple, set)) and allowed else None
            _stats["n"] += 1
            if _stats["n"] % 15 == 1:
                log.info(f"{label} PAYLOAD keys={sorted(data.keys())} teamId={data.get('teamId')!r} "
                         f"myPlayers={data.get('myPlayers')!r} gs_keys={sorted(gs.keys())} "
                         f"playMode={gs.get('playMode')!r} unlocked={sorted(allowed) if allowed else None} "
                         f"chat={str(gs.get('teamChat'))[:120]}")

            fast = enforce_allowed(tactics(gs, team_id, pid, shoot_now), allowed, gs, team_id, pid)
            if fast:
                if _stats["n"] % 10 == 0:
                    log.info(f"{label} RULE {fast[0]['commandType']} {int((time.time()-t0)*1000)}ms")
                yield json.dumps(fast)
                return

            summary = rich_summary(gs, team_id, pid, label)
            if allowed:
                summary += "\nALLOWED COMMANDS (use ONLY these): " + ", ".join(sorted(allowed))
            try:
                text = chain.call(summary, timeout)
                cmds = sanitize(parse_commands(text, team_id, pid), gs, team_id, pid, fallback_fn)
            except Exception as e:
                log.warn(f"{label} LLM unavailable ({type(e).__name__}) -> fallback")
                cmds = []
            if not cmds:
                cmds = fallback_fn(gs, team_id, pid)
            cmds = enforce_allowed(cmds, allowed, gs, team_id, pid)
            if cmds:
                log.info(f"{label} LLM returned {cmds[0]['commandType']} via {chain.current} "
                         f"{int((time.time()-t0)*1000)}ms")
                yield json.dumps(cmds[:1])
            else:
                yield json.dumps(enforce_allowed(fallback_fn(gs, team_id, pid), allowed, gs, team_id, pid))
        except Exception as e:
            log.error(f"{label} agent error: {e}")
            try:
                yield json.dumps(enforce_allowed(fallback_fn(gs, team_id, pid), allowed, gs, team_id, pid))
            except Exception:
                c = dict(last_resort)
                c["teamId"] = team_id
                yield json.dumps([c])

    return invoke
