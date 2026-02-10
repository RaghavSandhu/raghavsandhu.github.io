#!/bin/bash
# EC2 setup script for Stock Trend Predictor
# Run on a fresh Amazon Linux 2023 or Ubuntu 22.04 instance
#
# Usage:
#   chmod +x deploy/setup.sh
#   sudo ./deploy/setup.sh
#
# After setup:
#   1. Set ANTHROPIC_API_KEY in /etc/stock-predictor/env (optional)
#   2. sudo systemctl start stock-scheduler
#   3. sudo systemctl start stock-dashboard

set -euo pipefail

APP_DIR="/opt/stock-predictor"
APP_USER="stockapp"
REPO_URL="https://github.com/RaghavSandhu/raghavsandhu.github.io.git"
BRANCH="claude/setup-python-environment-Hr7HS"  # Change to "main" after merge

echo "=== Stock Trend Predictor — EC2 Setup ==="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi

# Install system dependencies
echo "[1/6] Installing system packages..."
if [ "$OS" = "amzn" ] || [ "$OS" = "rhel" ]; then
    dnf install -y python3.11 python3.11-pip git
elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3-pip git
else
    echo "Unsupported OS: $OS. Install Python 3.11+ manually."
    exit 1
fi

# Create app user
echo "[2/6] Creating app user..."
id -u $APP_USER &>/dev/null || useradd -r -m -s /bin/bash $APP_USER

# Clone/update repo
echo "[3/6] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
else
    git clone -b $BRANCH "$REPO_URL" "$APP_DIR"
fi
chown -R $APP_USER:$APP_USER "$APP_DIR"

# Install Python dependencies
echo "[4/6] Installing Python dependencies..."
cd "$APP_DIR"
pip3 install -r requirements.txt

# Create data directory
mkdir -p "$APP_DIR/data"
chown -R $APP_USER:$APP_USER "$APP_DIR/data"

# Create env file
echo "[5/6] Creating config..."
mkdir -p /etc/stock-predictor
cat > /etc/stock-predictor/env << 'ENVEOF'
# Set your Anthropic API key for LLM-powered analysis (optional)
# ANTHROPIC_API_KEY=sk-ant-...

PYTHONPATH=/opt/stock-predictor
ENVEOF

# Install systemd services
echo "[6/6] Installing systemd services..."

cat > /etc/systemd/system/stock-scheduler.service << 'EOF'
[Unit]
Description=Stock Prediction Batch Scheduler
After=network.target

[Service]
Type=simple
User=stockapp
WorkingDirectory=/opt/stock-predictor
EnvironmentFile=/etc/stock-predictor/env
ExecStart=/usr/bin/python3 -m src.batch.scheduler
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/stock-dashboard.service << 'EOF'
[Unit]
Description=Stock Prediction Streamlit Dashboard
After=network.target stock-scheduler.service

[Service]
Type=simple
User=stockapp
WorkingDirectory=/opt/stock-predictor
EnvironmentFile=/etc/stock-predictor/env
ExecStart=/usr/bin/python3 -m streamlit run src/dashboard/batch_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable stock-scheduler stock-dashboard

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. (Optional) Edit /etc/stock-predictor/env to add ANTHROPIC_API_KEY"
echo "  2. Start services:"
echo "     sudo systemctl start stock-scheduler"
echo "     sudo systemctl start stock-dashboard"
echo "  3. Open port 8501 in your EC2 security group"
echo "  4. Access dashboard at http://<your-ec2-ip>:8501"
echo ""
echo "Useful commands:"
echo "  sudo journalctl -u stock-scheduler -f    # Watch scheduler logs"
echo "  sudo journalctl -u stock-dashboard -f    # Watch dashboard logs"
echo "  sudo systemctl status stock-scheduler    # Check scheduler status"
echo "  sudo systemctl stop stock-scheduler      # Stop scheduler"
