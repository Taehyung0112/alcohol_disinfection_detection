import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

def is_finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y  # 指尖比中節高 => 展開

def is_hand_open_strict(hand_landmarks):
    fingers_extended = 0
    # INDEX, MIDDLE, RING, PINKY
    finger_pairs = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]
    for tip, pip in finger_pairs:
        if is_finger_extended(hand_landmarks.landmark, tip, pip):
            fingers_extended += 1

    return fingers_extended >= 3  # 至少三指展開才算張開

def detect_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape
    valid_hands_xy = []

    if results.multi_hand_landmarks:
        print(f"🖐️ Detected {len(results.multi_hand_landmarks)} hand(s)")
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_MCP]

            wx, wy = int(wrist.x * w), int(wrist.y * h)

            # 掌心方向
            v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
            v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
            palm_normal = np.cross(v1, v2)
            palm_normal_unit = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
            dot = np.dot(palm_normal_unit, np.array([0, 0, -1]))

            # 條件：掌心朝上 + 手指張開
            if dot > 0.5 and is_hand_open_strict(hand_landmarks):
                print(f"✅ Hand {i} upward (dot={dot:.2f}) and OPEN at ({wx},{wy})")
                valid_hands_xy.append((wx, wy))
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (wx, wy), 6, (0, 255, 0), -1)
            else:
                print(f"❌ Hand {i} rejected (dot={dot:.2f}, open={is_hand_open_strict(hand_landmarks)})")

    return valid_hands_xy
