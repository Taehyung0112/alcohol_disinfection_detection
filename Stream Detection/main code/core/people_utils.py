import cv2
from ultralytics.engine.results import Results  # 加這行以使用 YOLOv8 結果類型

def draw_people_boxes(results_people: list[Results], frame):
    people_boxes = {}
    for r in results_people:
        if not hasattr(r, "boxes") or r.boxes is None or r.boxes.id is None:
            continue
        for box, track_id_tensor in zip(r.boxes, r.boxes.id):
            track_id = int(track_id_tensor.item())
            cls_id = int(box.cls.item())
            label = r.names[cls_id]
            if label != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people_boxes[track_id] = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return people_boxes
