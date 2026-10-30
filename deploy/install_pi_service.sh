#!/usr/bin/env bash
# ==============================================================================
# Presento - Raspberry Pi Camera On-Demand Services Installer
# Configures:
#   1. presento-control.service (HTTP control daemon on port 5002 - ENABLED on boot)
#   2. presento-camera.service  (C270 TCP stream on port 5001 - ON-DEMAND, DISABLED on boot)
#   3. /etc/presento/control.env (Shared PRESENTO_CONTROL_TOKEN authentication)
#   4. /etc/sudoers.d/presento-camera (Passwordless control for admin user)
# ==============================================================================
set -e

echo "========================================================"
echo "  Presento Pi Camera Services - On-Demand Installer     "
echo "========================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Verify required scripts in /home/admin/
if [ ! -f "/home/admin/camera_stream.py" ]; then
    echo "[!] /home/admin/camera_stream.py not found. Copying from repo..."
    if [ -f "$SCRIPT_DIR/../edge/camera_stream.py" ]; then
        cp "$SCRIPT_DIR/../edge/camera_stream.py" /home/admin/camera_stream.py
        chmod +x /home/admin/camera_stream.py
    fi
fi

echo "[+] Setting up /home/admin/presento_control.py..."
if [ -f "$SCRIPT_DIR/presento-control.py" ]; then
    cp "$SCRIPT_DIR/presento-control.py" /home/admin/presento_control.py
    chmod +x /home/admin/presento_control.py
elif [ -f "$SCRIPT_DIR/../edge/pi_control_server.py" ]; then
    cp "$SCRIPT_DIR/../edge/pi_control_server.py" /home/admin/presento_control.py
    chmod +x /home/admin/presento_control.py
fi

# 2. Setup /etc/presento directory and control token
sudo mkdir -p /etc/presento
sudo chmod 750 /etc/presento

CONTROL_ENV="/etc/presento/control.env"
if [ ! -f "$CONTROL_ENV" ]; then
    echo "[+] Generating secure PRESENTO_CONTROL_TOKEN..."
    GENERATED_TOKEN=$(openssl rand -hex 16 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(16))")
    echo "PRESENTO_CONTROL_TOKEN=$GENERATED_TOKEN" | sudo tee "$CONTROL_ENV" > /dev/null
    sudo chmod 640 "$CONTROL_ENV"
    sudo chown root:admin "$CONTROL_ENV" 2>/dev/null || sudo chown root:root "$CONTROL_ENV"
    echo "[+] Token generated in $CONTROL_ENV"
else
    echo "[i] Existing $CONTROL_ENV found. Preserving token."
    GENERATED_TOKEN=$(grep -E "^PRESENTO_CONTROL_TOKEN=" "$CONTROL_ENV" | cut -d'=' -f2- | tr -d '"'"'")
fi

# 3. Setup sudoers rule so admin can start/stop camera service without password
echo "[+] Configuring sudoers rule for camera service control..."
SUDOERS_FILE="/etc/sudoers.d/presento-camera"
echo "admin ALL=(ALL) NOPASSWD: /bin/systemctl start presento-camera.service, /bin/systemctl stop presento-camera.service, /bin/systemctl restart presento-camera.service, /bin/systemctl status presento-camera.service, /bin/systemctl is-active presento-camera.service" | sudo tee "$SUDOERS_FILE" > /dev/null
sudo chmod 440 "$SUDOERS_FILE"

# 4. Copy systemd service units
echo "[+] Installing systemd service units..."
if [ -f "$SCRIPT_DIR/presento-control.service" ]; then
    sudo cp "$SCRIPT_DIR/presento-control.service" /etc/systemd/system/presento-control.service
    sudo chmod 644 /etc/systemd/system/presento-control.service
fi

if [ -f "$SCRIPT_DIR/presento-camera.service" ]; then
    sudo cp "$SCRIPT_DIR/presento-camera.service" /etc/systemd/system/presento-camera.service
    sudo chmod 644 /etc/systemd/system/presento-camera.service
fi

# 5. Reload systemd daemon
echo "[+] Reloading systemd configuration..."
sudo systemctl daemon-reload

# 6. Enable Control Service on boot & start it
echo "[+] Enabling and starting presento-control.service (Port 5002)..."
sudo systemctl enable presento-control.service
sudo systemctl restart presento-control.service

# 7. CRITICAL: Disable Camera Service on boot & ensure it is stopped (ON-DEMAND)
echo "[+] Ensuring presento-camera.service is DISABLED on boot (On-Demand Mode)..."
sudo systemctl disable presento-camera.service || true
echo "[+] Ensuring camera is stopped initially..."
sudo systemctl stop presento-camera.service || true

# 8. Display Status Summary
echo ""
echo "========================================================"
echo "  SUCCESS: Presento On-Demand Services Configured!      "
echo "========================================================"
echo "  - Control Daemon : ACTIVE on port 5002 (starts at boot)"
echo "  - Camera Stream  : STOPPED / ON-DEMAND (disabled at boot)"
echo "  - Camera C270    : RELEASED (LED is OFF)"
echo ""
echo "  >>> COPY THIS TOKEN TO YOUR WINDOWS .env FILE <<<"
echo "  PRESENTO_CONTROL_TOKEN=$GENERATED_TOKEN"
echo "  PRESENTO_CONTROL_PORT=5002"
echo "  ================================================="
echo ""
echo "  Verification & Manual Commands:"
echo "    Check Control Service : sudo systemctl status presento-control"
echo "    Check Camera Service  : sudo systemctl status presento-camera"
echo "    Manual Start Camera   : sudo systemctl start presento-camera"
echo "    Manual Stop Camera    : sudo systemctl stop presento-camera"
echo "    View Logs             : journalctl -u presento-control -f"
echo "========================================================"
