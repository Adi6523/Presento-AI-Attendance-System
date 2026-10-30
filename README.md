# Presento AI Vision — Portable Smart Attendance & Face Recognition System

Presento is a state-of-the-art, privacy-preserving AI facial recognition and anti-spoofing attendance system engineered for high-throughput edge deployment. It seamlessly adapts across diverse Windows client machines—from integrated graphics ultrabooks to multi-GPU gaming laptops—as well as dedicated Raspberry Pi 4B edge camera stations.

---

## 1. What Presento Is

Presento automates classroom and enterprise attendance through non-intrusive, multi-model computer vision:
- **Zero-Touch Attendance**: Detects, recognizes, and confirms enrolled identities in real time at ~30 FPS.
- **Robust Anti-Spoof Protection**: Dual neural networks evaluate planar distortion, frequency texture, and depth cues to reject photos, screen replays, and video attacks.
- **Closed-Set Verification**: Unenrolled faces are securely quarantined as `UNKNOWN` rather than forced to the nearest candidate.
- **Hardware Agnostic & Portable**: Automatically inspects host CPU and GPU hardware at launch, selects the best verified inference backend, and never crashes on unsupported hardware.
- **Dual Operating Modes**:
  - **Standalone Laptop Mode**: Operates completely offline using the built-in laptop webcam.
  - **Raspberry Pi Network Mode**: Streams high-efficiency H.264 video from an edge-mounted Logitech C270 camera over TCP with on-demand hardware control.

---

## 2. Architecture & Pipeline

Presento uses an asynchronous, decoupled pipeline to maintain fluid 30 FPS display smoothness even on CPU-constrained hardware:

```
[ Camera Source ]
  (Laptop Webcam via OpenCV OR Raspberry Pi C270 via TCP 5001)
         │
         ▼
[ Frame Receiver / Thread ]  (Latest-frame buffer / drops stale backlog)
         │
         ▼
[ SCRFD Face Detector ]      (500M parameter ONNX model, 5 facial landmarks)
         │
         ▼
[ ArcFace Embedder ]         (MobileFaceNet 512-D normalized vector space)
         │
         ▼
[ Closed-Set Face Matcher ]  (Cosine distance, margin gate, temporal stability)
         │
         ▼
[ Dual Anti-Spoof Ensembles ]
   ├── MobileNetV3 (MN3)
   └── MiniFASNet V2 SE
         │
         ▼
[ 4-Gate Attendance Tracker ]
   Gate 1: Identity Confirmed
   Gate 2: Anti-Spoof PASS (REAL)
   Gate 3: Temporal Stability (5/5 frames)
   Gate 4: Deduplication Cooldown (10s)
         │
         ▼
[ Demo HUD & Attendance Log ]
```

---

## 3. Supported Hardware & Intelligent Selection

Presento dynamically analyzes your system architecture without requiring vendor-specific GPU driver SDKs (e.g., CUDA or ROCm):

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **CPU** | Any x86_64 CPU (Intel Core i3 / AMD Ryzen 3, 4 cores) | 11th Gen+ Intel Core i5/i7 or AMD Ryzen 5000+ (6+ cores) |
| **RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **GPU** | Standard Display Adapter (CPU Inference) | Intel Iris Xe / Intel UHD / Intel Arc (OpenVINO GPU Acceleration) |
| **OS** | Windows 10/11 (64-bit) | Windows 11 (64-bit) |
| **Camera** | Built-in Laptop Webcam (720p/1080p) | Logitech C270 HD Webcam (via Raspberry Pi 4B) |

### Hardware Detection Matrix:
- **Intel Integrated GPU (Iris Xe / UHD / HD Graphics)**: Automatically verified via the OpenVINO GPU runtime plugin. When verified, all four AI models are compiled directly to the Intel GPU for accelerated throughput.
- **NVIDIA Discrete GPU (GeForce RTX / GTX / Laptop GPU)**: Presento detects the GPU model and VRAM, but does **NOT** falsely claim NVIDIA GPU acceleration without a dedicated CUDA/TensorRT runtime. It conservatively selects **OpenVINO CPU**, cleanly explaining the decision to the user.
- **AMD Integrated / Discrete GPU (Radeon 780M / 680M / RX Series)**: Presento accurately discovers the device and safely routes inference to **OpenVINO CPU**.
- **Multi-GPU Laptops (e.g. Intel iGPU + NVIDIA dGPU or AMD iGPU + NVIDIA dGPU)**: Inspects and displays all installed graphics adapters, then picks the verified acceleration device (Intel iGPU) or falls back cleanly to CPU without guessing.
- **CPU-Only Desktops / Virtual Machines**: Operates smoothly on multi-core CPU using OpenVINO CPU AVX2/AVX-512 vectorization.

---

## 4. Startup Hardware Analysis Example

When starting Presento on an Intel laptop, the console displays:

```
============================================================
 PRESENTO AI VISION -- HARDWARE ANALYSIS
============================================================

 CPU
 ------------------------------------------------------------
 Vendor       : Intel
 Model        : 12th Gen Intel Core i5-1240P @ 2110 MHz
 Architecture : x86_64
 Cores        : 12 physical / 16 logical

 GPU
 ------------------------------------------------------------
 GPU 0        : Intel Iris Xe Graphics (1024 MB VRAM)
 Type         : Integrated
 Vendor       : Intel
 Driver       : 31.0.101.5590
 Status       : VERIFIED

 PRESENTO INFERENCE
 ------------------------------------------------------------
 Selected     : Intel Iris Xe Graphics
 Backend      : OpenVINO
 Device       : GPU
 Status       : VERIFIED
 Reason       : Intel GPU verified with native OpenVINO acceleration

 AI PIPELINE
 ------------------------------------------------------------
 SCRFD        : GPU
 ArcFace      : GPU
 Anti-Spoof   : GPU
 MiniFASNet   : GPU

============================================================
```

On an AMD / NVIDIA laptop:

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
 Vendor       : AMD
 Status       : DETECTED
 GPU 1        : NVIDIA GeForce RTX 4060 Laptop GPU (6144 MB VRAM)
 Type         : Discrete
 Vendor       : NVIDIA
 Driver       : 551.23
 Status       : DETECTED

 PRESENTO INFERENCE
 ------------------------------------------------------------
 Selected     : AMD Ryzen 7 7840HS
 Backend      : OpenVINO
 Device       : CPU
 Status       : READY
 Reason       : NVIDIA GPU detected; no verified Presento GPU backend configured

 AI PIPELINE
 ------------------------------------------------------------
 SCRFD        : CPU
 ArcFace      : CPU
 Anti-Spoof   : CPU
 MiniFASNet   : CPU

============================================================
```

---

## 5. Quick Start (Friend Testing)

### Step 1: Environment Setup
1. Ensure **Python 3.10–3.13** is installed and added to your system `PATH`.
2. Open PowerShell or Command Prompt in the project folder:
   ```cmd
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

### Step 2: Run the Demo
Double-click **`Start-Demo.bat`** (or run from terminal):
```
======================================================================
                       PRESENTO AI VISION DEMO
======================================================================

  1. Laptop Webcam Demo (Recommended for Friends / Offline)
  2. Raspberry Pi Camera Demo (Network Stream / Hardware Setup)
  Q. Quit

======================================================================
Enter your choice (1, 2, or Q): 1
```
Choose `1`. Presento will inspect your hardware, verify the inference device, initialize your webcam, and open the interactive demonstration HUD.

### Step 3: Interactive Hotkeys
| Key | Action |
| :---: | :--- |
| `[1]` | **Mode 1: Face Detection** (SCRFD bounding boxes + 5 landmarks) |
| `[2]` | **Mode 2: Face Recognition** (ArcFace embeddings + closed-set matching) |
| `[3]` | **Mode 3: Anti-Spoofing** (Dual MN3 + MiniFASNet liveness verification) |
| `[4]` | **Mode 4: Attendance Verification** (4-gate attendance confirmation) |
| `[5]` | **Mode 5: Summary Dashboard** (Real-time analytics and logs) |
| `[E]` | **Enroll Unknown Face** (Interactive in-demo face enrollment) |
| `[R]` | **Reset History** (Clears temporal identity smoothing buffer) |
| `[P]` | **Toggle Pipeline Architecture Panel** |
| `[D]` | **Hardware Diagnostics Dump** (Prints full hardware tree & FPS metrics to console) |
| `[Space]` | **Freeze / Resume Video** |
| `[Q]` / `[ESC]` | **Quit & Clean Release** |

---

## 6. Enrolling Students

You can enroll new students in two easy ways:

### Option A: Via GUI During Demo (`[E]` Key)
1. In Mode 2, position the person's face in the camera view until `[UNKNOWN]` appears.
2. Press `[E]`.
3. In the console, enter the Student ID (e.g. `DEMO001`) and Name.
4. The system immediately captures and computes the 512-D ArcFace embedding, stores it in `data/face_embeddings.npz`, and hot-reloads the matcher. The person is recognized instantly!

### Option B: Smart Guided Enrollment Tool (`Enroll-Student.bat`)
Run **`Enroll-Student.bat`**:
1. Enter Student ID and Student Name.
2. Select Camera (`[1]` Laptop Webcam or `[2]` Raspberry Pi Camera).
3. Follow the on-screen pose prompts (*Frontal, Turn Left, Turn Right, Look Up, Look Down, Smile, Distance*).
4. Automated filters reject blurry frames (Laplacian variance $< 35$) and duplicate vectors ($> 0.985$ cosine similarity).
5. Press `[S]` to save reference vectors.

---

## 7. Model Verification

Presento requires 4 pre-trained neural networks in `data/models/`:
1. `scrfd_500m_bnkps.onnx` — Ultra-lightweight face detector with 5 facial landmarks.
2. `w600k_mbf.onnx` — MobileFaceNet ArcFace feature extractor (512-D embedding space).
3. `anti-spoof-mn3.onnx` — MobileNetV3 multi-scale texture anti-spoof classifier.
4. `minifasnet_v2se.onnx` — MiniFASNet V2 SE frequency-domain liveness classifier.

At startup, Presento validates the presence of each model. If any model file is missing or corrupted, it displays a clear diagnostic error message with remediation steps rather than an unhandled Python traceback.

---

## 8. Raspberry Pi Edge Mode (Optional)

For deploying with an edge camera station:
1. Connect a **Raspberry Pi 4B** and a **Logitech C270 USB webcam** to your local Wi-Fi or hotspot.
2. On the Pi, install services using `deploy/install_pi_service.sh`.
3. Launch `Start-Demo.bat` and choose `[2] Raspberry Pi Camera Demo`.
4. Presento automatically discovers the Pi via mDNS (`presento-pi.local`) or `.env` configuration, turns on the camera service on-demand (`POST /camera/start`), and streams frames over port 5001.
5. When stopping (`[Q]`), the remote camera is cleanly shut down and the Logitech C270 LED turns OFF.

---

## 9. Troubleshooting & FAQ

- **"No camera frame received from webcam"**:
  Ensure no other app (e.g., Zoom, Teams, Windows Camera app) is using your webcam. If your laptop has multiple cameras, specify `--source 1` or set `CAMERA_INDEX=1` in `.env`.
- **"Why is OpenVINO CPU selected when I have an NVIDIA GPU?"**:
  Presento's inference engine natively targets OpenVINO. While NVIDIA GPUs are detected, Presento will never falsely claim NVIDIA acceleration unless a tested, hardware-verified backend is configured. OpenVINO's optimized CPU execution with AVX-512/AVX2 provides fast, smooth performance (~25-30 FPS) without requiring multi-gigabyte CUDA installations.
- **"Pytest verification"**:
  Run `pytest -q` from the root directory. All 214 unit and integration tests validate hardware detection, classification, fallback logic, and anti-spoof gating with 100% pass rate.
