import os
import cv2
import time
from ultralytics import YOLO
from mediapipe_utils.hand_detector import detect_hands_and_draw
from core.dispenser_detection import detect_dispenser
from core.people_utils import draw_people_boxes
from core.tracking_logic import (
    compute_disinfection_zone,
    draw_disinfection_zone,
    detect_triggered_hand,
    find_closest_person,
    update_sanitization_state
)

# === 設定相對路徑 ===
YOLO_DISPENSER_MODEL = os.path.join("..", "dispensor weight", "best(yolov11).pt")
YOLO_PEOPLE_MODEL = os.path.join("..", "dispensor weight", "yolo11n.pt")

# === 載入 YOLO 模型 ===
model_people = YOLO(YOLO_PEOPLE_MODEL)
model_dispenser = YOLO(YOLO_DISPENSER_MODEL)

# === 初始化變數 ===
dispenser_roi = None
sanitized_count = 0
sanitized_ids = set()
track_state = {}
DELAY_TIME = 3
DISPENSER_DETECT_FRAMES = 100
dispenser_detected = False
frame_count = 0

# === 開啟攝影機 ===
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 影像擷取失敗")
        break

    start_time = time.time()
    h, w, _ = frame.shape

    if not dispenser_detected and frame_count < DISPENSER_DETECT_FRAMES:
        roi_candidate = detect_dispenser(frame, model_dispenser)
        if roi_candidate:
            dispenser_roi = roi_candidate
            dispenser_detected = True
            print("Dispenser position fixed:", dispenser_roi)

    if dispenser_roi:
        results_people = model_people.track(frame, persist=True, tracker="botsort.yaml")
        hands_xy = detect_hands_and_draw(frame, w, h, dot_threshold=0.4)
        people_boxes = draw_people_boxes(results_people, frame)
        zone = compute_disinfection_zone(dispenser_roi)
        frame = draw_disinfection_zone(frame, zone)

        triggered_hand = detect_triggered_hand(hands_xy, zone)
        if triggered_hand and people_boxes:
            target_id = find_closest_person(people_boxes, triggered_hand)
            if target_id is not None:
                last_time = track_state.get(target_id, {}).get('last_detected', 0)
                if (time.time() - last_time) > DELAY_TIME:
                    sanitized_count = update_sanitization_state(
                        frame, target_id, time.time(), sanitized_ids, track_state,
                        DELAY_TIME, sanitized_count
                    )

    # === 顯示效能資訊 ===
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Sanitized Count: {sanitized_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Real-Time Sanitization Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()