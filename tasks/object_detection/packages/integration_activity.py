from typing import Tuple

# Path to the trained model weights (.onnx file).
# Relative paths resolve from the project root.
MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    # 0 = run detection every frame.
    # For real bot, you can later try 1 or 2 if it becomes slow.
    return 0


def filter_by_classes(pred_class: int) -> bool:
    """
    Classes:
        0 = duckie
        1 = truck
        2 = sign
    """
    return pred_class in (0, 1, 2)


def filter_by_scores(score: float) -> bool:
    """
    Generic score filter.
    Class-specific score filtering is also done inside agent.py.
    """
    return score >= 0.20


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Generic bbox filter.
    Class-specific duck/truck/sign filters are inside agent.py.
    """
    xmin, ymin, xmax, ymax = bbox

    box_w = xmax - xmin
    box_h = ymax - ymin

    if box_w <= 3 or box_h <= 3:
        return False

    area = box_w * box_h

    return area > 80
