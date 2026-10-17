"""DCON DEF agent - player 1. Hybrid prompt + reflexes + time-boxed model chain."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from fallback import build_fallback, DEF_CONFIG
from dcon_prompts import build_prompt
from dcon_core import build_handler

app = BedrockAgentCoreApp()
MY_PLAYER_ID = 1
POSITION_LABEL = "DEF"
SYSTEM_PROMPT = build_prompt(MY_PLAYER_ID)
fallback_commands = build_fallback(DEF_CONFIG)
build_handler(app, POSITION_LABEL, MY_PLAYER_ID, SYSTEM_PROMPT, fallback_commands, DEF_CONFIG, shoot_now=0)

if __name__ == "__main__":
    app.run()
