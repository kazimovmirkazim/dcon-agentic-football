"""Real Bedrock latency test with the DCON prompt + rich summary. Usage: python3 bench_dcon.py MODEL [MODEL...]"""
import sys, time, boto3, statistics as st, copy
sys.path.insert(0, 'lib')
from test_helpers import GAME_STATE
from dcon_prompts import build_prompt
from dcon_core import rich_summary
from parsing import parse_commands
br = boto3.client('bedrock-runtime', region_name='us-east-1')
gs = copy.deepcopy(GAME_STATE); gs["teamChat"] = ["Ares, shoot early!"]
for m in sys.argv[1:]:
    for pid, lab in [(2, "MID"), (3, "FWD1")]:
        lat, ok, err = [], 0, ''
        sp, s = build_prompt(pid), rich_summary(gs, 0, pid, lab)
        for i in range(5):
            t = time.time()
            try:
                r = br.converse(modelId=m, system=[{'text': sp}], messages=[{'role': 'user', 'content': [{'text': s}]}],
                                inferenceConfig={'maxTokens': 90, 'temperature': 0.2})
                lat.append(time.time() - t); txt = r['output']['message']['content'][0]['text']
                ok += 1 if parse_commands(txt, 0, pid) else 0
            except Exception as e:
                err = str(e)[:80]; break
        if lat:
            print(f"{m[3:]:42s} {lab:4s} p50={st.median(lat)*1000:5.0f}ms max={max(lat)*1000:5.0f}ms json={ok}/{len(lat)}  last: {txt[:70]}", flush=True)
        else:
            print(f"{m[3:]:42s} {lab:4s} XETA {err}", flush=True)
