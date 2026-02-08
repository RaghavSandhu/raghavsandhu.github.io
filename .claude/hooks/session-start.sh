#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies
pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt"

# Set PYTHONPATH so imports from src/ resolve correctly
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
