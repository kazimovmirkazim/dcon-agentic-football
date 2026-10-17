"""DCON FWD2 agent - player 4. Hybrid prompt + reflexes + time-boxed model chain."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from fallback import build_fallback, FWD2_CONFIG
from dcon_prompts import build_prompt
from dcon_core import build_handler

app = BedrockAgentCoreApp()
MY_PLAYER_ID = 4
POSITION_LABEL = "FWD2"
SYSTEM_PROMPT = build_prompt(MY_PLAYER_ID)
fallback_commands = build_fallback(FWD2_CONFIG)
build_handler(app, POSITION_LABEL, MY_PLAYER_ID, SYSTEM_PROMPT, fallback_commands, FWD2_CONFIG, shoot_now=24)

if __name__ == "__main__":
    app.run()
