#!/bin/bash
# Run from ~/sample-ai-possibilities/agentic-football-sample-agents
set -e
for a in gk def mid fwd1 fwd2; do
  cp ai-team-strands-balanced/ai-$a/.bedrock_agentcore.yaml.template ai-team-dcon/ai-$a/
  cp ai-team-strands-balanced/ai-$a/requirements.txt ai-team-dcon/ai-$a/
done
cp ai-team-strands-balanced/deploy-all.sh ai-team-dcon/ && chmod +x ai-team-dcon/deploy-all.sh
echo "DCON installed"
