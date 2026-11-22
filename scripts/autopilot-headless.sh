#!/bin/bash
# Run the Framewise autopilot on one episode unattended, in a headless Claude Code session that answers the
# second-reviewer questions between passes (docs/autopilot/HARNESS.md M8, skill .agents/skills/framewise-autopilot).
#   scripts/autopilot-headless.sh "Episode 6"
# The session uses the user's Claude plan (D6); the profile's Claude budget caps how many items it answers per run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EPISODE=${1:?name an episode, e.g. "Episode 6"}
cd "$ROOT"
LOG_DIR=${AUTOPILOT_LOG_DIR:-$HOME/.local/state/framewise-autopilot}
mkdir -p "$LOG_DIR"
exec claude -p "Use the framewise-autopilot skill to run the autopilot on the episode \"$EPISODE\": start an agent run, run fw autopilot --go, answer the items for Claude with the resolver skills, run fw autopilot --go again until it exits 0 or only the person's questions are left, finish the run and print the digest." \
  --allowedTools "Bash(scripts/fw:*),Bash(python3 scripts/fw:*),Read,Skill" >> "$LOG_DIR/$(date +%F)-$(echo "$EPISODE" | tr ' /' '__').log" 2>&1
