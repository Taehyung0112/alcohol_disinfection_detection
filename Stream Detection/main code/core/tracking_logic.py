import cv2

def compute_disinfection_zone(dispenser_roi, extend_ratio=0.5):
    height = dispenser_roi[3] - dispenser_roi[1]
    width = dispenser_roi[2] - dispenser_roi[0]

    # 左右各延伸 extend_ratio * width
    extend = int(width * extend_ratio)
    extended_x1 = max(0, dispenser_roi[0] - extend)
    extended_x2 = dispenser_roi[2] + extend

    lower_y1 = dispenser_roi[1] + int(height * 0.8)
    lower_y2 = dispenser_roi[3] + int(height * 0.3)

    return (extended_x1, lower_y1, extended_x2, lower_y2)


def draw_disinfection_zone(frame, zone):
    overlay = frame.copy()
    cv2.rectangle(overlay, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

def detect_triggered_hand(hands_xy, zone):
    for hand in hands_xy:
        if len(hand) >= 2:  # 容錯保險
            hx, hy = hand[0], hand[1]
            if zone[0] <= hx <= zone[2] and zone[1] <= hy <= zone[3]:
                print(f"✅ Hand entered disinfection zone at ({hx}, {hy})")
                return (hx, hy)
    return None


def find_closest_person(people_boxes, hand_pos):
    min_dist = float('inf')
    target_id = None
    for track_id, (x1, y1, x2, y2) in people_boxes.items():
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist = ((cx - hand_pos[0]) ** 2 + (cy - hand_pos[1]) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            target_id = track_id
    print(f"🎯 Closest person to hand: ID {target_id} (distance={min_dist:.2f})")
    return target_id

def update_sanitization_state(frame, target_id, current_time, sanitized_ids, track_state, delay_time, sanitized_count):
    if target_id not in sanitized_ids:
        sanitized_ids.add(target_id)
        sanitized_count += 1
        print(f"✅ Person ID {target_id} triggered disinfection")
    track_state[target_id] = {'last_detected': current_time, 'show_text': True}
    cv2.putText(frame, f"detected {target_id} Sanitization Success!", (frame.shape[1] // 4, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return sanitized_count