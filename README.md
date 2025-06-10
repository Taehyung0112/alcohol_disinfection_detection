# 🧼 Real-Time Alcohol Dispenser Hand Detection System

This project detects hand disinfection behavior using YOLO object detection and Mediapipe hand landmarks. It ensures that users sanitize their hands **with palms facing upward** under an alcohol dispenser. Optimized for Jetson Xavier and cross-platform desktop systems.

## 🔍 Features

- 🎯 Detects alcohol dispensers and nearby persons using YOLOv11
- 🖐️ Uses Mediapipe to detect hands and compute if **palm is facing upward**
- 📦 Supports both `.pt` and `.engine` (TensorRT) model formats
- ✅ Identifies successful disinfection by checking hand entry and palm orientation
- 🧠 Tracks multiple people using BoT-SORT
- 🖥️ Supports both **video file input** and **real-time webcam inference**
- 📈 Displays real-time **FPS** and **sanitization count**

## 📁 Project Structure

```
Stream Detection/
├── Input Video/                # Test input videos
├── Output Video/               # Processed output videos
├── dispensor weight/           # YOLO models (.pt / .engine)
├── main code/
│   ├── main.py                 # Entry point for real-time detection
│   ├── core/                   # Core detection & logic modules
│   │   ├── dispenser_detection.py
│   │   ├── people_utils.py
│   │   ├── tracking_logic.py
│   │   └── video_utils.py
│   └── mediapipe_utils/
│       └── hand_detector.py
```

## 🧪 Requirements

- Python 3.9 or higher (3.10+ recommended)
- Conda (recommended) or virtualenv

### Python Libraries

```bash
pip install opencv-python mediapipe ultralytics numpy
```

> On Jetson Xavier, additional setup is needed for TensorRT + PyTorch compatibility.

## 🖥️ Usage

### 1. Real-time webcam detection

```bash
python main.py
```

> Press `q` to exit the webcam viewer.

### 2. Optional: Inference on pre-recorded video

To process a specific video file:

```python
cap = cv2.VideoCapture("Input Video/test.mp4")
```

Replace in `main.py` or switch back to `initialize_video()` if you use the modular version.

## 🤚 Palm-Up Detection Logic

- Mediapipe provides 21 hand landmarks.
- We use **Wrist**, **Index MCP**, and **Pinky MCP** to compute a palm normal vector.
- If the **dot product** between the normal and `[0, 0, -1]` is greater than `0.3`, the palm is considered **facing upward**.

This helps ensure users are truly performing a disinfection gesture, not just passing their hand.

## 📦 Model Files

Place the following models in the `dispensor weight/` folder:

- `best(yolov11).pt` – Trained dispenser detection model
- `yolo11n.pt` – People detection model
- (Optional) `.engine` versions for TensorRT deployment

## 📊 Output Display

- Green boxes around persons with successful disinfection
- Yellow zone under dispenser = disinfection area
- Real-time FPS and count of sanitized individuals displayed on-screen

---

## 👨‍💻 Author

Developed by [楊哲綸]  
Contact: [oscar900112@gmail.com]
