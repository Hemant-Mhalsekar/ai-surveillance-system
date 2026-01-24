def is_inside_zone(bbox, zone):
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone

    # Use centroid to check entry
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return (zx1 <= cx <= zx2) and (zy1 <= cy <= zy2)


def detect_suspicious_events(detections, restricted_zone, max_people_allowed):
    events = []

    # Crowd alert
    if len(detections) > max_people_allowed:
        events.append("CROWD_ALERT")

    # Intrusion alert
    for d in detections:
        if is_inside_zone(d["bbox"], restricted_zone):
            events.append("INTRUSION")
            break

    return events
