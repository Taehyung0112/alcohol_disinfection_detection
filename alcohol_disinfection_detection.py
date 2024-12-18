#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
from ultralytics import YOLO
import mediapipe as mp
import time

# 設定路徑
YOLO_DISPENSER_MODEL = "/Users/yangzhelun/Desktop/國衛院專案/dispensor weight/best.pt" # 酒精機模型
VIDEO_INPUT = "/Users/yangzhelun/Downloads/IMG_7129.mp4"         # 輸入影片
VIDEO_OUTPUT = "/Users/yangzhelun/Desktop/國衛院專案/Output Video/processed_video.mp4"

# Mediapipe 初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# 初始化 YOLO 模型
model_people = YOLO("yolov8n.pt")
model_dispenser = YOLO(YOLO_DISPENSER_MODEL)

# 全域變數
dispenser_roi = None
sanitized_count = 0
sanitized_ids = set()
track_state = {}
INTERSECTION_THRESHOLD = 0.1
DELAY_TIME = 3


# 函數定義
def initialize_video(input_path, output_path):
    """ 初始化影片讀取和寫入 """
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return cap, out, fps, frame_width, frame_height


def detect_dispenser(frame):
    """ 偵測酒精機並返回位置 """
    results = model_dispenser(frame)
    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls)]
            confidence = box.conf[0]
            if label == 'dispenser' and confidence > 0.3:
                return tuple(map(int, box.xyxy[0]))  # 將 map 轉換為元組
    return None


def calculate_intersection_ratio(person_box, dispenser_roi):
    """ 計算人與酒精機的重疊比例 """
    def calculate_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    x1 = max(person_box[0], dispenser_roi[0])
    y1 = max(person_box[1], dispenser_roi[1])
    x2 = min(person_box[2], dispenser_roi[2])
    y2 = min(person_box[3], dispenser_roi[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection_area / min(calculate_area(person_box), calculate_area(dispenser_roi))


def process_hand_detection(frame, dispenser_roi, track_id, w, h):
    """ 使用 Mediapipe 檢測手部並確認是否進行消毒 """
    global sanitized_count
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lower_edge_start = dispenser_roi[1] + int((dispenser_roi[3] - dispenser_roi[1]) * 0.8)
            lower_edge_end = dispenser_roi[3]
            for lm in hand_landmarks.landmark:
                hand_x = int(lm.x * w)
                hand_y = int(lm.y * h)
                if dispenser_roi[0] <= hand_x <= dispenser_roi[2] and lower_edge_start <= hand_y <= lower_edge_end:
                    if track_id not in sanitized_ids:
                        sanitized_ids.add(track_id)
                        sanitized_count += 1
                        return True
    return False


def process_tracking(frame, results_people, dispenser_roi, w, h, current_time):
    """ 處理人物追蹤邏輯 """
    for r in results_people:
        for box, track_id_tensor in zip(r.boxes, r.boxes.id):
            track_id = int(track_id_tensor.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_box = (x1, y1, x2, y2)

            # 初始化追蹤狀態
            track_state.setdefault(track_id, {'last_detected': 0, 'show_text': False})

            # 計算重疊率
            overlap_ratio = calculate_intersection_ratio(person_box, dispenser_roi)

            # 手部檢測邏輯
            if overlap_ratio > INTERSECTION_THRESHOLD:
                if (current_time - track_state[track_id]['last_detected']) > DELAY_TIME:
                    if process_hand_detection(frame, dispenser_roi, track_id, w, h):
                        track_state[track_id] = {'last_detected': current_time, 'show_text': True}

            # 顯示提示文字
            if track_state[track_id]['show_text']:
                cv2.putText(frame, "Sanitization Success!", (w // 4, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # 繪製框與 ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
            if dispenser_roi is not None:
                overlap_ratio = calculate_intersection_ratio(person_box, dispenser_roi)


# 主程式
cap, out, fps, frame_width, frame_height = initialize_video(VIDEO_INPUT, VIDEO_OUTPUT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("處理完成")
        break

    current_time = time.time()
    h, w, _ = frame.shape

    if dispenser_roi is None:
        dispenser_roi = detect_dispenser(frame)
        continue

    results_people = model_people.track(frame, persist=True, tracker="botsort.yaml")
    process_tracking(frame, results_people, dispenser_roi, w, h, current_time)

    # 繪製消毒計數
    cv2.putText(frame, f"Sanitized Count: {sanitized_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Jetson Nano Sanitization Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

