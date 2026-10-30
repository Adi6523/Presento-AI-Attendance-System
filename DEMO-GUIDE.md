# Presento AI Vision — Demonstration Guide & Playbook
**Project Demonstration & Evaluation Guide**

---

## 1. Overview & System Goals

The **Presento AI Vision Demo** provides an interactive, visual demonstration overlay of the facial-recognition and multi-stage verification pipeline. It allows evaluators, professors, and peers to inspect the underlying computer vision and deep learning models operating in real time at **~30 FPS** on OpenVINO GPU (Intel Iris Xe) and Raspberry Pi 4B hardware.

### Live AI Vision Pipeline Flowchart

```
       [1] Logitech C270 / Raspberry Pi Camera
                          ↓  (1280x720 TCP Stream, ZeroLag FrameReceiver)
       [2] SCRFD Lightweight Neural Detector (OpenVINO GPU)
                          ↓  (Bounding Box + 5 Facial Landmarks @ ~30 FPS)
       [3] ArcFace MobileFaceNet Embedder (OpenVINO GPU)
                          ↓  (512-Dimensional Normalized Feature Vector @ ~10-14 FPS)
       [4] Multi-Vector Face Matcher (Strict Closed-Set Policy)
                          ↓  (Cosine Similarity vs 0.48 Acceptance Threshold)
       [5] Multi-Frame Identity Stability Engine
                          ↓  (5-Frame Sliding Temporal Window)
       [6] Asynchronous Dual Anti-Spoof Fusion (GPU)
                          ↓  (Anti-Spoof-MN3 + MiniFASNet V2 SE @ 80% Consensus)
       [7] Attendance Gating & Async Network I/O
                          ↓  (Non-blocking background HTTP queue & sync)
                   [ PRESENT ]
```

---

## 2. Interactive Demo Screens / Modes

Switch modes at any time during the live feed using keys **`[1]`** through **`[5]`** (or press **`[M]`** / **`[TAB]`** to cycle):

| Mode | Title | Visual Focus & What to Explain |
| :--- | :--- | :--- |
| **`[1]`** | **Face Detection** | Focuses on **SCRFD** detection. Shows bounding box with corner brackets, 5 facial landmarks (Eyes, Nose, Mouth corners), inter-pupillary axis, and head tilt angle normalized from `-180°` to `+180°` (frontal is `~0.0°`). |
| **`[2]`** | **Face Recognition** | Focuses on **ArcFace** 512-D embeddings and **FaceMatcher**. Implements strict closed-set recognition: if similarity $< 0.48$, displays `UNKNOWN: No enrolled identity accepted`. If enrolled, displays name, student ID, cosine score, margin to 2nd candidate, and multi-frame stability. |
| **`[3]`** | **Anti-Spoof (Liveness)** | Focuses on presentation attack defense. Shows dual-model breakdown: **MN3** $p_{\text{real}}/p_{\text{spoof}}$ + **MiniFASNet V2 SE** $p_{\text{real}}/p_{\text{spoof}}$, inference latencies, 15-frame window consensus, and temporal status (`REAL`, `SPOOF`, or `UNCERTAIN`). |
| **`[4]`** | **Attendance Gating** | Focuses on the 4-gate verification checklist required before attendance is awarded: Enrolled Identity Match + Multi-Frame Stability + Anti-Spoof REAL + Cooldown Ready. Shows green `[PRESENT]` confirmation upon passing all gates. Blocks UNKNOWN or SPOOF. |
| **`[5]`** | **Summary Dashboard** | Displays an analytics dashboard over the live feed showing session uptime, display/detector/embedder FPS, total faces seen, recognized students, unknown presentations, blocked spoof attacks, and recent attendance log. |

### Keyboard Shortcuts Reference

- **`[1]` - `[5]`** : Jump directly to Mode 1, 2, 3, 4, or 5.
- **`[M]` / `[TAB]`** : Cycle to next mode.
- **`[P]`** : Toggle the **AI Pipeline Architecture Panel** on the right side of the screen.
- **`[E]`** : **Demo Enrollment Flow**: Trigger guided capture of an unknown person into the active session.
- **`[R]`** : **Retry Recognition**: Flushes stability history for instant re-testing.
- **`[Space]`** : Freeze / Pause frame (freeze current frame to explain landmarks or similarity scores). Press Space again to resume.
- **`[D]`** : Print a formatted diagnostic snapshot to the terminal console.
- **`[Q]` / `[ESC]`** : Cleanly stop the demo and release camera hardware.

---

## 3. Strict Closed-Set Recognition Architecture

### Core Design Rules
1. **Enrolled-Only Recognition**: The pretrained ArcFace model generates a 512-D embedding for any human face. However, the application's recognition policy matches **only against students enrolled in `data/face_embeddings.npz`**.
2. **Acceptance Threshold**: A face is recognized only if the cosine similarity between its embedding and an enrolled student's reference cluster satisfies:
   $$\text{similarity} \ge 0.48$$
3. **Honest UNKNOWN Display**: If similarity is below `0.48`, the system outputs:
   - `Identity: UNKNOWN`
   - `No enrolled identity accepted`
   - Nearest neighbor candidate is shown **strictly as diagnostic information** (`Best: <Candidate> (sim: 0.xx) - Acceptance: FAILED`) and **never** assigned as the person's identity.
4. **Attendance Gate**: An `UNKNOWN` face is strictly blocked from receiving attendance under all circumstances.

---

## 4. Step-by-Step Demonstration Walkthrough

### Step 1: How to Start the Demo
Double-click **`Start-Demo.bat`** (or run `.\start_demo.ps1`).
- The launcher probes `presento-pi.local:5002` and sends `POST /camera/start`.
- The Logitech C270 camera green LED turns **ON**.
- Real-time OpenVINO HUD displays at ~30 FPS.

### Step 2: How to Demonstrate an Enrolled Student
1. Have an enrolled student (e.g., **Demo Student** `DEMO001`) look into the camera.
2. In **Mode 2 (Face Recognition)**:
   - Bounding box turns **Emerald Green**.
   - State badge: **`[RECOGNIZED]`**.
   - Shows candidate name, student ID, similarity score (`~0.75 - 0.85`), margin, and stability (`5/5`).
3. In **Mode 4 (Attendance Gating)**:
   - All 4 verification gates turn **PASS**.
   - State changes to **`[PRESENT]`**.
   - Buzzer tone plays and non-blocking event is queued.

### Step 3: How to Demonstrate an Unknown Face
1. Have a non-enrolled person step in front of the camera.
2. In **Mode 2**:
   - Bounding box turns **Slate Gray**.
   - State badge: **`[UNKNOWN]`**.
   - Shows: `Identity: UNKNOWN`, `No enrolled identity accepted`.
   - Diagnostic line: `Best: Demo Student (0.34) - Acceptance: FAILED`.
3. In **Mode 4**:
   - Gate 1 (Identity Confirmed): **FAILED (UNKNOWN)**.
   - Attendance: **BLOCKED**.

### Step 4: Demo-Only Unknown Enrollment (`[E]` Key)
1. While an unknown face is on screen, press **`[E]`**.
2. The demo pauses live recognition and prompts in the console:
   ```
   Student ID: 23UAM999
   Student Name: Test Guest
   Confirm enrollment? [Y/N]: Y
   ```
3. The demo enters **Smart Guided Enrollment**:
   - Live camera guides through diverse poses: *Frontal, Turn Left, Turn Right, Look Up, Look Down, Smile, Distance variations*.
   - Built-in filters reject blurry frames (Laplacian variance $< 35$), small faces, and duplicates ($> 0.985$ cosine similarity).
   - Press **`[C]`** or **`[SPACE]`** to capture each pose.
   - Press **`[S]`** to save once 8–12 samples are captured.
4. The new embeddings are saved to `data/face_embeddings.npz` and the in-memory matcher is hot-reloaded.
5. Recognition resumes immediately: **the guest is now recognized!**

### Step 5: How to Demonstrate Anti-Spoof Protection
1. Switch to **Mode 3 (Anti-Spoof)**.
2. Real person facing camera:
   - Dual models (MN3 + MiniFASNet) evaluate texture, frequency, and depth cues.
   - Status: **`REAL`** (95–100% agreement).
3. Smartphone photo / screen attack:
   - Hold up a smartphone with a photo of an enrolled student.
   - Models detect 2D screen reflections, color distortion, and planar boundaries.
   - Status changes to **`SPOOF DETECTED`** or **`UNCERTAIN`**.
   - In Mode 4, attendance is **BLOCKED**.

### Step 6: How to Stop the Demo & Verify Camera Release
1. Press **`[Q]`** or **`[ESC]`**.
2. The demo releases FrameReceiver and triggers `POST /camera/stop`.
3. **Verify Camera Release**:
   - Check the physical Logitech C270 camera: **green LED is completely OFF**.
   - Run `curl http://presento-pi.local:5002/camera/status` → `{"service_active": false}`.

---

## 5. Demo Student Registry

In the portable demonstration package, the enrollment registry contains synthetic demonstration identities:

| # | Student Name | Student ID | Vector Samples | Status |
| :- | :--- | :--- | :-: | :--- |
| 1 | **Demo Student 1** | `DEMO001` | 5 | Synthetic Demo |
| 2 | **Demo Student 2** | `DEMO002` | 5 | Synthetic Demo |

*(No real personal photographs or real biometric data are included).*

To inspect the store at any time:
```powershell
.venv\Scripts\python.exe -m edge.verify_enrollment
```

---

## 6. Friend Test Checklist

```markdown
[ ] Project starts cleanly via Start-Demo.bat
[ ] Virtual environment (.venv) detected and validated
[ ] Pi camera control server discovered on port 5002
[ ] Camera starts ON DEMAND when demo or enrollment begins (C270 LED turns ON)
[ ] Camera stops completely after exit (C270 LED turns OFF)
[ ] Enrolled person recognized with correct name, ID, and similarity >= 0.75
[ ] Unknown person displays UNKNOWN and is NOT given any enrolled identity
[ ] Unknown person cannot receive attendance under any mode
[ ] Pressing [E] during live demo prompts for ID/Name with confirmation
[ ] Smart enrollment captures diverse poses (frontal, left, right, up, down)
[ ] Newly enrolled person is immediately recognized without restarting the app
[ ] Dual anti-spoof models (MN3 + MiniFASNet) report real face as REAL
[ ] Smartphone photo / screen attack triggers SPOOF or UNCERTAIN
[ ] Spoof / Uncertain strictly blocks attendance
[ ] Attendance event is recorded and queued asynchronously
[ ] Duplicate attendance within 60s cooldown is prevented
[ ] Mode 1 (Detection), Mode 2 (Recognition), Mode 3 (Anti-Spoof) work
[ ] Mode 4 (Gating) and Mode 5 (Dashboard) work
[ ] No continuous camera stream or background process remains after exit
[ ] Store verification tool confirms 4 core demo students intact
```
