import cv2

def detect_dispenser(frame, model_dispenser):
    results = model_dispenser(frame)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = r.names[cls_id]
            confidence = box.conf[0]
            if label.lower() == "dispenser" and confidence > 0.3:
                coords = list(map(int, box.xyxy[0]))
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 255), 2)
                cv2.putText(frame, "Dispenser", (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                print(f"\n✅ Dispenser detected at: {coords}")
                return coords
    return None