import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

def detect_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape
    hands_xy = []
    if results.multi_hand_landmarks:
        print(f"🖐️ Detected {len(results.multi_hand_landmarks)} hand(s)")
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * w)
            y = int(wrist.y * h)
            print(f"   ↳ Hand {i} wrist at ({x}, {y})")
            hands_xy.append((x, y))
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
    return hands_xy