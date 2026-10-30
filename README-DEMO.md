# Presento AI Vision — Demo & Friend Testing Guide
**Prepared for September 30, 2026 Project Presentation & External Testing**

Welcome to **Presento AI Vision**! This guide provides complete instructions for setting up, running, enrolling students, and testing the facial recognition and anti-spoofing demo on Windows with a connected Raspberry Pi 4B + Logitech C270 camera.

---

## 1. System Requirements

### Hardware Requirements
- **Host Laptop / PC**:
  - Windows 10 or 11 (64-bit)
  - Processor: Intel Core i5/i7 (8th Gen+) or modern AMD Ryzen
  - Integrated GPU: Intel Iris Xe or Intel UHD (OpenVINO GPU acceleration supported) or CPU fallback
  - RAM: 8 GB minimum (16 GB recommended)
  - Network: Wi-Fi or Ethernet connected to the **same local network / hotspot** as the Raspberry Pi
- **Edge Camera Station**:
  - Raspberry Pi 4B (2GB/4GB/8GB) running Raspberry Pi OS (Debian 64-bit)
  - Logitech C270 USB HD Webcam connected to a USB 2.0/3.0 port on the Pi
  - 5V/3A USB-C Power Supply for the Raspberry Pi
- **Alternative (No Pi available)**:
  - Any standard USB webcam plugged directly into the host laptop (`--source 0`).

---

## 2. Windows Setup & Dependencies

1. **Python Installation**:
   - Python 3.10, 3.11, 3.12, or 3.13 (64-bit).
   - Ensure **"Add Python to PATH"** was checked during installation.
2. **Git** (optional, for cloning):
   - Git for Windows.
3. **PowerShell**:
   - Built-in Windows PowerShell 5.1 or PowerShell 7+.

---

## 3. Python Virtual Environment Setup

From the root project directory in a command prompt or PowerShell:

```powershell
# 1. Create a dedicated virtual environment if not already created
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1
# (or in Command Prompt: .\.venv\Scripts\activate.bat)

# 3. Upgrade pip and install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

*Note: Pre-compiled OpenVINO runtime libraries (`openvino>=2024.0.0`) and OpenCV are installed via `requirements.txt`.*

---

## 4. Raspberry Pi Setup

The Raspberry Pi 4B runs an on-demand camera control service.

1. **Camera Architecture (On-Demand)**:
   - **Port 5002**: HTTP Control Server (`camera_server.py`) with endpoints:
     - `GET /health` / `GET /camera/status`
     - `POST /camera/start`
     - `POST /camera/stop`
   - **Port 5001**: TCP MJPEG Video Stream (active **only** during sessions).
   - The Logitech C270 camera **remains OFF when idle** (green LED is off).
2. **Starting the Pi Control Server** (on the Pi):
   ```bash
   cd ~/presento-edge
   python3 pi_service.py
   # Or enable the systemd service:
   # sudo systemctl start presento-camera.service
   ```

---

## 5. Raspberry Pi IP Discovery & Configuration

By default, the Windows software discovers the Pi automatically:
- Broadcasts/queries `presento-pi.local` or subnet scan on ports `5001` / `5002`.
- You can manually specify the Pi's IP address if mDNS/multicast is blocked on your router/hotspot:
  ```powershell
  $env:PI_IP = "192.168.1.105"
  ```
  Or pass it directly:
  ```powershell
  .\Start-Demo.bat
  # When prompted, type: 192.168.1.105
  ```

---

## 6. How to Start Presento (Full Stack)

If you wish to run the full Presento attendance ecosystem (Backend API + Database + Web Dashboard):

```powershell
# Start Presento Full Stack
.\start_presento.ps1
```
This initializes:
- SQLite Database: `data/presento.db` (seeded with demo course/student roster)
- Backend Fastify / Node API: `http://localhost:3000`
- Web Frontend: `http://localhost:5173`

---

## 7. How to Start the AI Vision Live Demo

Double-click **`Start-Demo.bat`** (or **`START-DEMO.bat`**) in the project folder.

Alternatively, from PowerShell:
```powershell
.\start_demo.ps1
```

### What happens at launch:
1. Checks virtual environment `.venv\`.
2. Probes the Raspberry Pi camera service on port 5002.
3. Automatically triggers `POST /camera/start` → **C270 green LED turns ON**.
4. FrameReceiver locks onto zero-lag TCP MJPEG stream (port 5001) at 1280x720.
5. SCRFD detector and ArcFace embedder compile on OpenVINO GPU (Intel Iris Xe).
6. Enrolled embeddings load from `data/face_embeddings.npz`.
7. OpenCV HUD window **"Presento AI Vision Demo"** displays at **~30 FPS**.

### Interactive Controls:
- **`[1]`** : **Mode 1 — Face Detection** (SCRFD bounding boxes + 5 landmarks + head tilt).
- **`[2]`** : **Mode 2 — Face Recognition** (ArcFace 512-D embeddings, cosine similarity, margin, stability).
- **`[3]`** : **Mode 3 — Anti-Spoof (Liveness)** (Dual-model MN3 + MiniFASNet V2 SE temporal fusion).
- **`[4]`** : **Mode 4 — Attendance Verification** (4-gate verification checklist: Identity + Stability + Liveness + Cooldown).
- **`[5]`** : **Mode 5 — Summary Dashboard** (Session metrics, face counters, spoof block stats, attendance log).
- **`[E]`** : **Enroll Unknown Person** (Guided interactive enrollment directly from the demo window).
- **`[R]`** : **Retry / Clear History** (Flushes multi-frame stability history for instant re-testing).
- **`[P]`** : Toggle the **AI Pipeline Architecture Panel** on the right side.
- **`[Space]`** : Freeze / Pause frame (inspect landmarks/scores); press again to resume.
- **`[D]`** : Dump diagnostic snapshot to console.
- **`[Q]` / `[ESC]`** : Stop demo and turn camera **OFF**.

---

## 8. How to Enroll a Student

There are two easy methods to enroll a student:

### Method A: Using `Enroll-Student.bat` (Recommended for New Students)
1. Double-click **`Enroll-Student.bat`**.
2. Enter the Student ID (e.g., `23UAM099`) and Student Name (e.g., `Friend Name`).
3. The script automatically commands the Pi camera to turn ON.
4. An OpenCV window opens with **Smart Guided Enrollment**:
   - Follow the 12 guided pose prompts on screen:
     - *Frontal, Turn slight left (~15°), Turn slight right (~15°), Tilt up, Tilt down, Smile, Distance variations.*
   - Quality filtering rejects blurry frames (Laplacian variance), too-small faces, or missing landmarks.
   - Duplicate rejection prevents capturing identical frames.
   - Press **`[C]`** or **`[SPACE]`** to capture each pose.
   - Once 8–12 diverse captures are accepted, press **`[S]`** to save.
5. The embeddings are merged into `data/face_embeddings.npz`.
6. The camera automatically turns **OFF** upon completion.

### Method B: Live During Demo (`[E]` Key)
1. In the live demo, have an unenrolled person look into the camera.
2. The HUD displays **`UNKNOWN: No enrolled identity accepted`**.
3. Press **`[E]`** on your keyboard.
4. In the console, enter the Student ID and Name, then confirm with `Y`.
5. The demo guides you through diverse pose captures using the active OpenVINO pipeline.
6. Once saved, recognition resumes immediately — the person is now recognized!

---

## 9. How to Test an Enrolled Face

1. Start the demo (`Start-Demo.bat`).
2. Have an enrolled student (e.g., **Demo Student**, `DEMO001`) face the camera.
3. Switch to **Mode 2** (`[2]`):
   - Bounding box turns **Emerald Green**.
   - Identity: `Demo Student`
   - Student ID: `DEMO001`
   - Similarity: `~0.75 - 0.85` (well above the `0.48` acceptance criteria).
   - Margin: `+0.45 - +0.60` (distance to 2nd closest student).
   - Stability: `5/5` frames confirmed.
4. Switch to **Mode 4** (`[4]`):
   - All 4 verification gates turn **PASS**.
   - Status transitions to **`[PRESENT]`**.
   - Attendance event is queued asynchronously.

---

## 10. How to Test an Unknown Face (Closed-Set Recognition)

1. Have a person who is **NOT** enrolled stand in front of the camera.
2. In **Mode 2** (`[2]`):
   - Bounding box turns **Slate Gray**.
   - State badge: **`[UNKNOWN]`**
   - Candidate line: `Identity: UNKNOWN`
   - Explanatory note: `No enrolled identity accepted`
   - Closest candidate diagnostic: `Best: <Student> (sim < 0.48) - Acceptance: FAILED`
3. In **Mode 4** (`[4]`):
   - Gate 1 (Identity Confirmed): **FAILED (UNKNOWN)**.
   - Attendance status: **BLOCKED**.
   - No attendance is ever generated for an unknown face.

---

## 11. How to Test Phone / Photo Anti-Spoofing

1. Switch to **Mode 3** (`[3]`) and press **`[P]`** to open the Pipeline Panel.
2. **Real Face Test**:
   - Real person looks at camera.
   - Both MN3 and MiniFASNet V2 SE report high $p_{\text{real}}$ (`> 0.90`).
   - Temporal fusion displays **`REAL`** (bright green).
3. **Photo / Screen Spoof Attack Test**:
   - Hold up a smartphone or tablet displaying a clear photo of an enrolled student, or a printed photo.
   - The dual anti-spoof models detect 2D planar textures, specular reflection, and boundary edges.
   - Temporal fusion flags **`SPOOF DETECTED`** (bright red) or **`UNCERTAIN`** (amber).
   - Switch to **Mode 4**: Attendance is strictly **BLOCKED**.

---

## 12. How to Stop the Demo

1. Press **`[Q]`** or **`[ESC]`** inside the OpenCV video window.
2. The cleanup routine:
   - Flushes pending attendance I/O events.
   - Shuts down worker threads.
   - Disconnects TCP video stream.
   - Sends `POST /camera/stop` to the Raspberry Pi control server.
   - Destroys OpenCV windows.

---

## 13. How to Verify the Logitech C270 is OFF Afterward

Verify camera shutdown using any of these checks:
1. **Physical LED Check**: Look at the Logitech C270 webcam on the Pi. The green activity indicator LED must be **completely OFF**.
2. **Terminal Confirmation**: The console prints:
   ```
   Camera session released and Pi camera service turned OFF.
   ```
3. **HTTP Query Check**: Run in PowerShell:
   ```powershell
   curl http://presento-pi.local:5002/camera/status
   ```
   Expected response:
   ```json
   {"service_active": false, "active_session": null}
   ```

---

## 14. Known Limitations & Engineering Notes

1. **Pretrained Generic Embedder**: ArcFace extracts 512-D geometric features from any face. The application's closed-set policy enforces recognition only for vectors matching enrolled students ($\text{similarity} \ge 0.48$).
2. **Passive 2D Anti-Spoofing**: Lightweight neural anti-spoof models run in real time on edge hardware without specialized infrared (IR) or depth sensors. High-refresh OLED displays under certain room lighting may produce an `UNCERTAIN` classification; in Presento, `UNCERTAIN` safely blocks attendance.
3. **Wi-Fi Bandwidth**: The 1280x720 30 FPS stream requires a stable 5 GHz Wi-Fi or wired Ethernet connection between the Pi and Windows laptop for optimal latency.

---

## 15. Friend Test Checklist

Use this checklist when testing the deployment ZIP with peers:

```markdown
[ ] 1. Project starts cleanly without dependency errors
[ ] 2. Virtual environment (.venv) detected and validated
[ ] 3. Pi camera control server discovered on port 5002
[ ] 4. Camera starts ON DEMAND when demo or enrollment begins (C270 LED turns ON)
[ ] 5. Camera stops completely after exit (C270 LED turns OFF)
[ ] 6. Enrolled person recognized with correct name, ID, and similarity >= 0.75
[ ] 7. Unknown person displays UNKNOWN and is NOT given any enrolled identity
[ ] 8. Unknown person cannot receive attendance under any mode
[ ] 9. Pressing [E] during live demo prompts for ID/Name with confirmation
[ ] 10. Smart enrollment captures diverse poses (frontal, left, right, up, down)
[ ] 11. Newly enrolled person is immediately recognized without restarting the app
[ ] 12. Dual anti-spoof models (MN3 + MiniFASNet) report real face as REAL
[ ] 13. Smartphone photo / screen attack triggers SPOOF or UNCERTAIN
[ ] 14. Spoof / Uncertain strictly blocks attendance
[ ] 15. Attendance event is recorded and queued asynchronously
[ ] 16. Duplicate attendance within 60s cooldown is prevented
[ ] 17. Mode 1 (Detection), Mode 2 (Recognition), Mode 3 (Anti-Spoof) work
[ ] 18. Mode 4 (Gating) and Mode 5 (Dashboard) work
[ ] 19. No continuous camera stream or background process remains after exit
[ ] 20. Store verification tool confirms 4 core demo students intact
```
