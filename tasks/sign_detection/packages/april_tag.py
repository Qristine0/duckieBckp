import numpy as np
import cv2



def detect_tags(signBehavior, frame_rgb: np.ndarray) -> list:
        gray   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        params = signBehavior.cfg.camera_params
        try:
            tags = (
                signBehavior._detector.detect(
                    gray, estimate_tag_pose=True,
                    camera_params=params, tag_size=signBehavior.cfg.tag_size_m,
                ) if params else signBehavior._detector.detect(gray)
            )
        except Exception as e:
            print(f"[SignBehavior] AprilTag error: {e}")
            return []

        filtered = [t for t in tags if t.decision_margin >= signBehavior.cfg.min_margin]
        if filtered:
            print(f"[SignBehavior] tags visible: {[t.tag_id for t in filtered]}")
        return filtered
    

def confirm_tags(signBehavior, raw_tags: list):
    seen_ids = {t.tag_id for t in raw_tags}
    for k in [k for k in signBehavior._tag_buffer if k not in seen_ids]:
        del signBehavior._tag_buffer[k]
    for tid in seen_ids:
        signBehavior._tag_buffer[tid] = signBehavior._tag_buffer.get(tid, 0) + 1
    confirmed = [tid for tid, cnt in signBehavior._tag_buffer.items()
                if cnt >= signBehavior.cfg.tag_confirm_frames]
    if confirmed:
        print(f"[SignBehavior] confirmed tags: {confirmed}")
    return confirmed
