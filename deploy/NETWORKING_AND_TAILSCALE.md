# Presento Pi ↔ Windows Connectivity & Networking Playbook

This document details the automated connectivity architecture between the **Raspberry Pi 4B (Camera Streamer)** and the **Windows Laptop (AI Recognition Runner)**.

---

## 1. Systemd Service Setup on Raspberry Pi

The camera stream must start automatically when the Raspberry Pi boots and restart if interrupted, without requiring an SSH session or manual command execution.

### Installation Steps on the Raspberry Pi:

1. **SSH into the Raspberry Pi (one-time setup)**:
   ```bash
   ssh admin@presento-pi.local
   # or: ssh admin@<CURRENT_PI_IP>
   ```

2. **Verify `camera_stream.py` location**:
   Ensure `camera_stream.py` is present at `/home/admin/camera_stream.py`.

3. **Install the systemd service unit**:
   Create `/etc/systemd/system/presento-camera.service`:
   ```bash
   sudo nano /etc/systemd/system/presento-camera.service
   ```
   Paste the following unit configuration:
   ```ini
   [Unit]
   Description=Presento Raspberry Pi Camera TCP Stream Service (C270)
   After=network-online.target time-sync.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=admin
   WorkingDirectory=/home/admin
   ExecStart=/usr/bin/python3 /home/admin/camera_stream.py --device /dev/video0 --host 0.0.0.0 --port 5001 --quality 85
   Restart=always
   RestartSec=3

   # Logging & resource safety
   StandardOutput=journal
   StandardError=journal
   SyslogIdentifier=presento-camera
   Environment=PYTHONUNBUFFERED=1

   TimeoutStopSec=5
   KillSignal=SIGTERM

   [Install]
   WantedBy=multi-user.target
   ```

4. **Reload, Enable, and Start the Service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable presento-camera.service
   sudo systemctl start presento-camera.service
   ```

5. **Verify Service Status**:
   ```bash
   sudo systemctl status presento-camera.service
   ```
   You should see: `Active: active (running)` listening on `0.0.0.0:5001`.

6. **Useful Maintenance Commands on the Pi**:
   ```bash
   # View live streaming logs:
   journalctl -u presento-camera -f

   # Restart camera service:
   sudo systemctl restart presento-camera

   # Stop camera service:
   sudo systemctl stop presento-camera
   ```

---

## 2. Windows Discovery & Zero-Config Connectivity

Windows discovery is handled automatically by [edge/discovery.py](file:///e:/College/4th%20year/Sem%20I/Caostone%20Project/edge/discovery.py) and [edge/frame_receiver.py](file:///e:/College/4th%20year/Sem%20I/Caostone%20Project/edge/frame_receiver.py).

### How It Works:
1. **mDNS Resolution (`presento-pi.local`)**:
   - Raspberry Pi OS runs `avahi-daemon` by default, broadcasting its hostname `presento-pi` via Multicast DNS on `224.0.0.251:5353`.
   - Windows 10/11 includes native mDNS resolution in its DNS client.
   - When Windows connects to `presento-pi.local:5001`, Windows automatically queries the local subnet for the Pi's current IP address.
2. **Dynamic Re-resolution & DHCP Tolerance**:
   - Every reconnection attempt re-queries `resolve_hostname("presento-pi.local")`.
   - If the Pi reboots and DHCP assigns it a new IP address, Windows automatically connects to the new IP address without any manual `.env` changes!
3. **Candidate Rotation & Offline Handling**:
   - If the Pi is powered off, Windows `FrameReceiver` logs the status and backs off (1.5s → 2.2s → 3.3s → 5.0s max).
   - As soon as the Pi powers on and `presento-camera.service` starts, Windows connects automatically.
   - No crash occurs while the Pi is offline.

---

## 3. Network Limitations: Same Wi-Fi vs Different Wi-Fi

### A. Same Wi-Fi / Local Area Network (Supported Natively)
- **Mechanism**: Direct TCP + mDNS (`presento-pi.local`).
- **Condition**: Both the Raspberry Pi and the Windows laptop are connected to the same Wi-Fi network, mobile hotspot, or Ethernet switch.
- **Result**: Immediate, zero-config connection. No special software required.

### B. Different Wi-Fi Networks (Cannot Connect Directly)
- **Technical Explanation**:
  1. **Non-routable RFC 1918 Private IPs**: A Wi-Fi network assigns private IPs (e.g. `192.168.1.x` or `172.31.x.x`). These addresses only exist inside that local router's LAN and are dropped by Internet routers.
  2. **NAT / Firewall Blocks Inbound Traffic**: Home and campus routers block incoming unsolicited connections from the Internet.
  3. **mDNS Is Link-Local Only**: Multicast DNS (`.local`) has a Time-To-Live (TTL) of 1. Routers strictly drop mDNS packets at the network boundary.
  4. **Port Forwarding Warning**: Opening port 5001 on a public router is insecure (unencrypted raw video stream exposed to the internet) and usually impossible on college/cellular networks due to Carrier-Grade NAT (CGNAT).

---

## 4. Cross-Network Solution: Tailscale Overlay Network

If you need the Raspberry Pi and Windows laptop to communicate across **different Wi-Fi networks** (e.g. Pi is on a mobile hotspot, Windows laptop is on college Wi-Fi), **Tailscale** is the recommended solution.

### Why Tailscale is Ideal:
- **Zero Port Forwarding**: Uses WireGuard encrypted UDP tunnels and NAT traversal (DERP relays).
- **End-to-End Encryption**: Camera stream is encrypted using ChaCha20-Poly1305.
- **Fixed Overlay IP & MagicDNS**: The Pi gets a permanent address (e.g. `100.x.y.z`) and domain (`presento-pi`).
- **LAN Optimization**: If both devices happen to be on the same Wi-Fi, Tailscale automatically establishes direct peer-to-peer connection for zero added latency.

### Quick Tailscale Setup:

1. **Install Tailscale on Raspberry Pi**:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
   Follow the terminal authentication URL to link the Pi to your Tailscale account.

2. **Install Tailscale on Windows**:
   - Download and install [Tailscale for Windows](https://tailscale.com/download/windows).
   - Log into the same Tailscale account.

3. **Configure Presento for Tailscale**:
   In [`.env`](file:///e:/College/4th%20year/Sem%20I/Caostone%20Project/.env), simply set:
   ```bash
   PI_IP=presento-pi
   # or the Pi's Tailscale 100.x.y.z IP address
   ```
   Because `edge/discovery.py` already includes `presento-pi` in its candidate list, Presento connects seamlessly whether on local Wi-Fi or over Tailscale!

---

## 5. Starting Presento on Windows

You do NOT need to open multiple terminals manually.

### One-Click Launch Options:
- **Batch File**: Double-click [`Start-Presento.bat`](file:///e:/College/4th%20year/Sem%20I/Caostone%20Project/Start-Presento.bat)
- **PowerShell**: Run [`start_presento.ps1`](file:///e:/College/4th%20year/Sem%20I/Caostone%20Project/start_presento.ps1)

### What the Launcher Does:
1. Runs a preflight check on Python `.venv`, demo accounts, and dependencies.
2. Performs instant discovery probe for the Pi camera stream.
3. Spawns the **Presento API** (`http://127.0.0.1:8000`).
4. Spawns the **Presento Dashboard** (`http://localhost:5173`).
5. Spawns the **Presento Camera Recognition Runner** (auto-connecting to `presento-pi.local:5001`).
6. Automatically opens your browser to the dashboard.
