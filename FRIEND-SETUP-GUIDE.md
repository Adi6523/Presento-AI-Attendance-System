# Presento AI Vision — Friend Setup & Testing Guide

Welcome! Thank you for helping test **Presento AI Vision**.

This guide is designed for anyone testing the project on their personal Windows laptop. You do **not** need an expensive GPU, specialized AI hardware, or a Raspberry Pi. Presento is engineered to adapt automatically to almost any standard Windows laptop with a built-in webcam.

---

## 1. What is Presento?

Presento is a portable, edge-optimized AI computer vision application for classroom attendance and identity verification. It combines four deep learning neural networks into an integrated, real-time pipeline:

1. **SCRFD Face Detection**: Detects faces in every frame and locates 5 facial landmark keypoints (eyes, nose, mouth corners).
2. **ArcFace Face Recognition**: Extracts a compact, normalized 512-dimensional biometric embedding representing facial geometry.
3. **Dual Anti-Spoof Ensembles (MN3 + MiniFASNet)**: Evaluates texture, planar distortion, and reflection cues to distinguish real human faces from screen replays, smartphone photos, or printed pictures.
4. **4-Gate Attendance Verification**: Confirms recognized identity, verifies liveness, checks multi-frame stability, and logs attendance while preventing duplicate entries.

---

## 2. What You Need

- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Camera**: Standard built-in laptop webcam or any plug-and-play USB webcam.
- **Python**: Python 3.10, 3.11, 3.12, or 3.13 installed on your machine.
- **Hardware**: Any Intel or AMD processor (4+ cores recommended, 8 GB RAM).
- **Internet Connection**: Only needed for 2 minutes to download standard Python dependencies via `pip`. Once installed, Presento runs **100% offline**.

---

## 3. Extracting the ZIP

1. Right-click `Presento-AI-Vision-Portable.zip` and choose **Extract All...**
2. Extract the folder to a convenient path on your PC (for example, `C:\Presento\` or your Desktop).
3. Open the extracted folder `Presento-AI-Vision-Portable`. Inside, you will see folders like `edge/`, `backend/`, `data/models/`, and batch scripts like `Start-Demo.bat`.

---

## 4. Checking Python

Make sure Python is installed and accessible in your command prompt:

1. Press `Win + X` and select **Terminal** or **PowerShell** (or search for `PowerShell` in the Start menu).
2. Type the following command and press Enter:

```powershell
python --version
```

- If you see `Python 3.10.x`, `Python 3.11.x`, `Python 3.12.x`, or `Python 3.13.x`, you are ready!
- If you see an error like `'python' is not recognized`:
  - Download Python from [python.org/downloads](https://www.python.org/downloads/).
  - **Important**: During installation, check the box that says **"Add python.exe to PATH"**.

---

## 5. Creating the Virtual Environment

In your PowerShell window, navigate to the extracted folder:

```powershell
cd path\to\Presento-AI-Vision-Portable
```

Create an isolated virtual environment named `.venv`:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

*(Note: If PowerShell displays a script execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and try activating again, or use standard Command Prompt and run `.\.venv\Scripts\activate.bat`)*.

Once activated, your prompt will show `(.venv)` on the left.

---

## 6. Installing Requirements

Ensure `pip` is up to date, then install the lightweight project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs OpenVINO, OpenCV, NumPy, and basic utilities. It does **not** install multi-gigabyte proprietary GPU SDKs like CUDA or ROCm.

---

## 7. Starting Presento

Simply run:

```cmd
Start-Demo.bat
```

You will see the main interactive launcher menu:

```
======================================================================
                        PRESENTO AI VISION DEMO
======================================================================

  1. Laptop Webcam Demo (Recommended for Friends / Offline)
  2. Raspberry Pi Camera Demo (Network Stream / Hardware Setup)
  Q. Quit

======================================================================
Enter your choice (1, 2, or Q):
```

### Choose `1` for the Laptop Webcam Demo:
- Runs **completely locally and offline**.
- Does **not** contact any Raspberry Pi.
- Does **not** require network ports 5001 or 5002.
- Uses your laptop webcam.
- Automatically inspects your machine's hardware and selects the best verified inference backend.

---

## 8. Automatic Hardware Detection

At startup, Presento scans your system hardware and prints an analysis box:

```
============================================================
 PRESENTO AI VISION -- HARDWARE ANALYSIS
============================================================

 CPU
 ------------------------------------------------------------
 Vendor       : Intel
 Model        : Intel Core i5-1240P
 Architecture : x86_64
 Cores        : 12 physical / 16 logical

 GPU
 ------------------------------------------------------------
 GPU 0        : Intel Iris Xe Graphics (1024 MB VRAM)
 Type         : Integrated
 Vendor       : Intel
 Status       : VERIFIED

 PRESENTO INFERENCE
 ------------------------------------------------------------
 Selected     : Intel Iris Xe Graphics
 Backend      : OpenVINO
 Device       : GPU
 Status       : VERIFIED
 Reason       : Intel GPU verified with native OpenVINO acceleration
============================================================
```

### What if you have an NVIDIA or AMD laptop?
If your laptop has an NVIDIA GeForce / RTX GPU or an AMD Radeon GPU, Presento will accurately detect and list the GPU:

```
============================================================
 PRESENTO AI VISION -- HARDWARE ANALYSIS
============================================================

 CPU
 ------------------------------------------------------------
 Vendor       : AMD
 Model        : AMD Ryzen 7 7840HS
 Architecture : x86_64
 Cores        : 8 physical / 16 logical

 GPU
 ------------------------------------------------------------
 GPU 0        : AMD Radeon 780M
 Type         : Integrated
 GPU 1        : NVIDIA GeForce RTX 4060 Laptop GPU
 Type         : Discrete
 Status       : DETECTED

 PRESENTO INFERENCE
 ------------------------------------------------------------
 Selected     : AMD Ryzen 7 7840HS
 Backend      : OpenVINO
 Device       : CPU
 Status       : READY
 Reason       : NVIDIA GPU detected; no verified Presento GPU backend configured
============================================================
```

**Why CPU?**  
Presento is conservative and truthful: it will **never falsely claim** GPU acceleration unless the selected runtime has verified and compiled models to that GPU. OpenVINO utilizes advanced AVX2 / AVX-512 CPU vectorization, providing smooth 25–30 FPS performance directly on your processor without requiring you to install complex CUDA/ROCm drivers.

---

## 9. Demo Modes

The demo window provides 5 progressive operational views:

- **Mode 1: Face Detection**  
  Displays raw SCRFD detection boxes with confidence scores and 5 cyan landmark dots (eyes, nose tip, mouth corners).
- **Mode 2: Face Recognition (Closed-Set)**  
  Matches faces against the enrolled database. Enrolled faces display emerald green with name, student ID, similarity score, and stability. Unknown faces display slate gray with `[UNKNOWN]`.
- **Mode 3: Anti-Spoof Protection**  
  Displays real-time liveness analysis from dual neural networks (MobileNetV3 + MiniFASNet) with liveness probability and classification (`REAL`, `SPOOF`, or `UNCERTAIN`).
- **Mode 4: Attendance Verification**  
  Evaluates the 4 security verification gates (Identity, Anti-Spoof, Stability, Cooldown). When all 4 gates pass, attendance transitions to `[PRESENT]` and plays an audible confirmation tone.
- **Mode 5: Summary Dashboard**  
  Shows an analytics HUD with total faces detected, recognized encounters, spoof rejections, and recent verified attendance event logs.

---

## 10. Keyboard Controls

Press these keys anytime while the demo window is focused:

| Key | Function | Description |
| :---: | :--- | :--- |
| **`[1]`** | Mode 1 | Face Detection (SCRFD + 5 Landmarks) |
| **`[2]`** | Mode 2 | Face Recognition (ArcFace + Closed-Set Matcher) |
| **`[3]`** | Mode 3 | Anti-Spoofing (Dual MN3 + MiniFASNet) |
| **`[4]`** | Mode 4 | Attendance Verification (4-Gate Confirmation) |
| **`[5]`** | Mode 5 | Summary Dashboard (Metrics & Logs) |
| **`[P]`** | Architecture Panel | Toggles the real-time AI Pipeline architecture flow diagram |
| **`[E]`** | Enroll Unknown | Enrolls an unknown face directly inside the demo |
| **`[R]`** | Reset History | Resets identity temporal smoothing buffer |
| **`[Space]`** | Freeze / Pause | Pauses/unpauses video feed |
| **`[D]`** | Diagnostics | Dumps full hardware tree and FPS metrics to the terminal |
| **`[Q]` / `[ESC]`** | Quit | Cleanly releases webcam and exits |

---

## 11. Enrolling Yourself

To test face recognition with your own face:

### Method A: In-GUI Quick Enrollment (`[E]` Key)
1. Run `Start-Demo.bat` and select `1`.
2. Press **`[2]`** to enter Mode 2.
3. Look directly at your camera until the bounding box shows `[UNKNOWN]`.
4. Press the **`[E]`** key on your keyboard.
5. In the console window, enter your details:
   - Student ID: (e.g. `TEST001` or your name initials)
   - Student Name: (e.g. `Your Name`)
6. Press Enter. Presento normalizes your 512-D ArcFace embedding, saves it to `data/face_embeddings.npz`, and hot-reloads the matcher.
7. Return to the demo window: **you will now be recognized instantly by name!**

### Method B: Guided Multi-Pose Tool (`Enroll-Student.bat`)
1. Double-click `Enroll-Student.bat`.
2. Enter your Student ID and Name.
3. Select `[1]` for Laptop Webcam.
4. Follow the on-screen pose prompts (*Look straight, turn slightly left, turn slightly right, tilt up, tilt down, smile*).
5. Press **`[C]`** or **`[Space]`** to capture each pose.
6. Once 8–12 diverse samples are captured, press **`[S]`** to save.

---

## 12. Testing Face Recognition

1. With the demo running, switch to **Mode 2** (`[2]`).
2. Position your face in front of the camera.
3. Verify:
   - The bounding box turns **Emerald Green**.
   - Your name and student ID appear above the box.
   - The similarity score displays `~0.75 - 0.85` (acceptance threshold is `0.48`).
   - The stability badge confirms `5/5` frames.

---

## 13. Testing an Unknown Person

1. Have a friend or family member who has **not** enrolled step in front of the camera (or look away from the camera).
2. Verify:
   - The bounding box turns **Slate Gray**.
   - The badge displays **`[UNKNOWN]`**.
   - The system displays: `"Identity: UNKNOWN - No enrolled identity accepted"`.
   - The system refuses to match an unknown face to any enrolled student.

---

## 14. Testing Anti-Spoofing

Switch to **Mode 3** (`[3]`) to test liveness verification:

1. **Real Human Face**:
   - Look naturally at the camera.
   - Both MN3 and MiniFASNet evaluate micro-textures and depth.
   - Status badge displays **`REAL`** (typically 90–100% agreement).
2. **Smartphone Screen Attack**:
   - Take a clear photo of yourself on your smartphone.
   - Hold the smartphone screen up to your laptop webcam.
   - The dual neural networks detect 2D screen reflections, pixel moiré patterns, and planar boundaries.
   - Status transitions to **`SPOOF DETECTED`** or **`UNCERTAIN`**.
   - Switch to **Mode 4**: notice that Attendance Gate 2 fails, and attendance is **BLOCKED**.
3. **Understanding `UNCERTAIN`**:
   - Anti-spoofing models can occasionally return `UNCERTAIN` if lighting is too dim, if the face is too far from the camera, or if glare obscures the face.
   - `UNCERTAIN` is an intentional security design: the system requires a definitive `REAL` result before granting attendance.

---

## 15. Hardware Diagnostics

Press **`[D]`** at any point during the demo. Look at your terminal console: Presento will dump a comprehensive hardware diagnostics tree:

```
============================================================
 PRESENTO HARDWARE & AI RUNTIME DIAGNOSTICS
============================================================

CPU
+-- Vendor: AMD
+-- Model: AMD Ryzen 7 7840HS
+-- Architecture: x86_64
+-- Physical cores: 8
\-- Logical processors: 16

GPU
+-- GPU 0: AMD Radeon 780M
|   +-- Vendor: AMD
|   +-- Type: Integrated
|   \-- Runtime status: Detected (Inference on CPU)
\-- GPU 1: NVIDIA GeForce RTX 4060 Laptop GPU
    +-- Vendor: NVIDIA
    +-- Type: Discrete
    \-- Runtime status: Detected (Inference on CPU)

AI RUNTIME
+-- OpenVINO version: 2024.x
+-- Selected device: CPU
+-- Device type: CPU
\-- Backend verification: READY

PERFORMANCE
+-- Display FPS: 29.8
+-- Detection FPS: 28.5
+-- ArcFace FPS: 25.0
\-- Anti-Spoof latency: 32.0ms
============================================================
```

---

## 16. Raspberry Pi Mode (Optional)

You do **not** need a Raspberry Pi for normal friend testing. However, if you or your lab has a Raspberry Pi 4B with a Logitech C270 camera:
1. Ensure the Pi and your laptop are on the same Wi-Fi network.
2. In `Start-Demo.bat`, select `[2] Raspberry Pi Camera Demo`.
3. Presento will automatically discover the Pi via mDNS (`presento-pi.local`), remotely turn on the camera on-demand via HTTP (`port 5002`), and stream high-speed frames over TCP (`port 5001`).
4. When you press `[Q]`, Presento remotely releases the camera and turns off the Logitech C270 green LED.

---

## 17. Common Problems & Troubleshooting

- **`'python' is not recognized`**:
  Reinstall Python from [python.org](https://www.python.org/downloads/) and ensure **"Add python.exe to PATH"** is checked.
- **PowerShell script execution disabled**:
  Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in PowerShell, or use standard Command Prompt (`cmd.exe`).
- **Webcam error (`[ERROR] Could not open camera`)**:
  - Close any apps that might be using the webcam (Zoom, Microsoft Teams, Discord, Windows Camera app, browser tabs).
  - If your laptop has multiple cameras (or an external webcam), you can pass `--source 1` or edit `CAMERA_INDEX=1` in `.env.example`.
- **"GPU detected but CPU selected"**:
  This is completely normal and expected for NVIDIA, AMD, or machines without verified Intel OpenVINO GPU drivers. Presento runs smoothly on CPU.
- **"Required AI model files are missing"**:
  Make sure all four `.onnx` files are located in `data/models/` (`scrfd_500m_bnkps.onnx`, `w600k_mbf.onnx`, `anti-spoof-mn3.onnx`, `minifasnet_v2se.onnx`). Do not delete or rename the `data/models/` folder.
- **Low FPS / Sluggish video**:
  If on CPU, close heavy background programs (e.g. video games, heavy browser tabs). Presento drops stale frames to preserve responsiveness, but 4+ CPU cores yield the smoothest experience.

---

## 18. Privacy Notes

- Presento processes all video frames **locally on your device**. No video, photos, or biometrics are ever uploaded to cloud servers.
- The shareable package contains **zero** real personal photos, zero personal attendance logs, and zero private credentials.
- When you enroll yourself, your facial features are stored solely as numerical mathematical vectors (float32 arrays) in `data/face_embeddings.npz` on your own laptop.

---

## 19. Quick Start Summary

1. **Extract** `Presento-AI-Vision-Portable.zip`.
2. **Open PowerShell** in the folder.
3. Run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Run:
   ```cmd
   Start-Demo.bat
   ```
5. Select **`[1] Laptop Webcam Demo`**.
6. Test Modes `1` through `5`.
7. Press **`[Q]`** to exit.

---

## 20. What to Send Back After Testing

After running the demo, please copy and fill out this short feedback form and message it back to us:

```text
--- PRESENTO TEST REPORT ---
Laptop Brand / Model   :
CPU                    :
GPU(s) Detected        :
GPU Type (iGPU/dGPU)   :
Selected Device        : (CPU or GPU)
Inference Backend      : OpenVINO
Approximate FPS        : (Shown in HUD top right)
Webcam opened cleanly  : (Yes / No)
Face detection worked  : (Yes / No)
Face recognition worked: (Yes / No)
Enrollment worked      : (Yes / No)
Anti-spoofing worked   : (Yes / No)
Attendance gate passed : (Yes / No)
Clean shutdown on [Q]  : (Yes / No)
Any problems / notes   :
----------------------------
```

*(Note: Please do **NOT** send any photos of yourself, biometric data files, passwords, or personal credentials).*
