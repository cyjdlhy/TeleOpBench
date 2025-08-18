# 📷 Camera-Based Human Motion Capture Pipeline

This document explains how to use our camera-based method to capture human motion and drive the TeleOpBench system. The pipeline consists of two components:

1. **Human motion capture via webcam using SMPLerX + MediaPipe**
2. **Motion control in TeleOpBench with real-time visualization**

---

## 🧍 Step 1: Human Motion Capture

We use a webcam to capture human motion, then process it using [SMPLerX](https://github.com/vchoutas/smplx) and [MediaPipe](https://github.com/google/mediapipe).

### 📦 Setup

Clone the repository:

```bash
git clone https://github.com/cyjdlhy/smpler_media.git
cd smpler_media
```

Install the required environment and download models by following instructions from the official SMPL-X repository:
👉 [https://github.com/vchoutas/smplx](https://github.com/vchoutas/smplx)

Then activate the environment:

```bash
conda activate smplerx
```

Install MediaPipe:

```bash
pip install mediapipe
```

### 🎥 Run Motion Capture

To start capturing motion via webcam:

```bash
cd main
bash run.sh
```

This will capture motion data (body pose + hand position) and send it for downstream processing.

---

## 🤖 Step 2: Motion Control in TeleOpBench

The motion data is received by TeleOpBench, which performs inverse kinematics (IK) and controls a robot accordingly, with real-time visualization.

### ⚙️ Setup

Activate the TeleOpBench environment:

```bash
conda activate teleop
```

Navigate to the teleop directory and run:

```bash
cd teleop
./run.sh
```

### 🕸️ Visualization

Once started, MeshCat will launch automatically, displaying real-time 3D visualization of the robot executing the motion.

---


