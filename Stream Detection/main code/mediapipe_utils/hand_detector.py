# === mediapipe_utils/hand_detector.py ===
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3)

def detect_hands_and_draw(frame, w, h):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hands_xy = []
    upward_hands = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            wx, wy = int(wrist.x * w), int(wrist.y * h)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (wx, wy), 6, (0, 0, 255), -1)

            # 判斷手掌是否朝上（容忍版本）
            v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
            v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
            palm_normal = np.cross(v1, v2)
            palm_normal_unit = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
            dot = np.dot(palm_normal_unit, np.array([0, 0, -1]))

            if dot > 0.3:
                upward_hands.append((wx, wy, dot))
                print(f"✅ Hand {i} upward (dot={dot:.2f}) at ({wx},{wy})")
            else:
                print(f"Hand {i} NOT upward (dot={dot:.2f})")

            hands_xy.append((wx, wy))
    return upward_hands
