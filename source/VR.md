# 🥽 VisionPro-Based Teleoperation Pipeline

This document explains how to use our VR-based method to capture human motion and drive the TeleOpBench system. The pipeline consists of two components:

1. **Hand motion capture via VR**
2. **Motion control in TeleOpBench with real-time visualization**

## 🧍 Step 1: Hand motion capture via VR

⚠️ **Make sure the Vision Pro and the computer are connected to the same local network (LAN)!**

1. check **Host machine** local ip address:

```bash
ifconfig | grep inet # Check the local IP address
```

2. install mkcert

   please refer to [https://github.com/FiloSottile/mkcert]
3. create certificate:

```bash
cd teleop
mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.123.1 localhost 127.0.0.1
```

4. open firewall on server:

```bash
sudo ufw allow 8012
```

5. install ca-certificates on Apple Vision Pro:

```bash
mkcert -CAROOT
```

Copy the `rootCA.pem` via AirDrop to Apple Vision Pro and install it.

Settings > General > About > Certificate Trust Settings. Under "Enable full trust for root certificates", turn on trust for the certificate.

> In the new version of Vision OS 2, this step is different: After copying the certificate to the Apple Vision Pro device via AirDrop, a certificate-related information section will appear below the account bar in the top left corner of the Settings app. Tap it to enable trust for the certificate.

Settings > Apps > Safari > Advanced > Feature Flags > Enable WebXR Related Features.


6. Open Safari on Apple Vision Pro and visit:

https://192.168.123.1:8012?ws=wss://192.168.123.1:8012

Make sure `192.168.123.1` is the IP address of your **Host machine**.

Click **Enter VR** and select **Allow** to start the VR session.



# 🤖 Step 2: Motion Control in TeleOpBench

### ⚙️ Setup

Activate the TeleOpBench environment:

```bash
conda activate teleop
```

Navigate to the teleop directory and run:

./run.sh

### 🕸️ Visualization

Once started, MeshCat will launch automatically, displaying real-time 3D visualization of the robot executing the motion.
