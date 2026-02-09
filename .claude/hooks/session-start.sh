#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies
pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt"

# Install PyTorch CPU (optional — models fall back to sklearn if unavailable)
pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true

# Set PYTHONPATH so imports from src/ resolve correctly
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
