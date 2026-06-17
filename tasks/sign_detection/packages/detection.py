# =========================================================
# GLOBAL CONFIG
# =========================================================
IMG_WIDTH = 640
IMG_HEIGHT = 480

CLASS_NAMES = {
    0: "duckie",
    1: "truck",
    2: "sign",
}

CLASS_COLORS = {
    0: (0, 215, 255),
    1: (180, 100, 220),
    2: (50, 205, 50),
}


vehicle_min_bbox_area = 800
# intersection crossing check
def vehicle_detected(detections):
    # Used by sign_behavior.py during CHECKPATH.
    if detections is None:
        detections = []

    for (x1, y1, x2, y2), score, cls_id in detections:
        if cls_id != 1:
            continue

        area = (x2 - x1) * (y2 - y1)

        if area < vehicle_min_bbox_area:
            continue

        centre_x = (x1 + x2) / 2.0
        offset = abs(centre_x - IMG_WIDTH / 2) / IMG_WIDTH

        print(f"[SignBehavior] vehicle detected (area={area:.0f}, offset={offset:.3f})")
        return True, offset

    return False, 0.0

