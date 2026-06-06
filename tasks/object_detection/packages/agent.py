import os
import warnings
import yaml
import numpy as np
import cv2
from typing import List, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

from . import integration_activity as student


def _find_config_file() -> str:
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
    )

    yaml_path = os.path.join(config_dir, "object_detection_config.yaml")
    yml_path = os.path.join(config_dir, "object_detection_config.yml")

    if os.path.exists(yaml_path):
        return yaml_path

    return yml_path


_CONFIG_FILE = _find_config_file()

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

CLASS_NAMES = {0: "duckie", 1: "truck", 2: "sign"}
CLASS_COLORS = {
    0: (0, 215, 255),
    1: (180, 100, 220),
    2: (50, 205, 50),
}

Detection = Tuple[Tuple[int, int, int, int], float, int]


def _xywh2xyxy(cx, cy, w, h, model_size, img_w, img_h):
    sx = img_w / model_size
    sy = img_h / model_size

    x1 = int((cx - w / 2) * sx)
    y1 = int((cy - h / 2) * sy)
    x2 = int((cx + w / 2) * sx)
    y2 = int((cy + h / 2) * sy)

    return (
        max(0, x1),
        max(0, y1),
        min(img_w - 1, x2),
        min(img_h - 1, y2),
    )


class ObjectDetectionAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE

        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[ObjectDetection] Config not found: {path}")
            cfg = {}
        except Exception as exc:
            print(f"[ObjectDetection] Could not load config: {exc}")
            cfg = {}

        # IMPORTANT: your best.onnx works at 640.
        self.img_size = cfg.get("img_size", 640)

        self.conf_threshold = cfg.get("conf_threshold", 0.25)
        self.nms_threshold = cfg.get("nms_threshold", 0.45)

        self.duck_conf_threshold = cfg.get("duck_conf_threshold", 0.30)
        self.truck_conf_threshold = cfg.get("truck_conf_threshold", 0.30)
        self.sign_conf_threshold = cfg.get("sign_conf_threshold", 0.35)

        self.duck_min_height_ratio = cfg.get("duck_min_height_ratio", 0.025)
        self.duck_min_area_ratio = cfg.get("duck_min_area_ratio", 0.0006)
        self.duck_max_width_height_ratio = cfg.get("duck_max_width_height_ratio", 2.60)
        self.duck_max_height_width_ratio = cfg.get("duck_max_height_width_ratio", 4.00)
        self.duck_ignore_top_ratio = cfg.get("duck_ignore_top_ratio", 0.20)

        self.model_path = self._resolve_model_path(student.MODEL_PATH)

        self.frame_count = 0
        self.session = None
        self.net = None
        self._backend = None

        self.model_loaded = False
        self.load_error = None

        self.trt_building = False
        self._trt_build_start = None

        self._last_detections: List[Detection] = []

        self._load_model()

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        if os.path.isabs(model_path):
            return model_path

        return os.path.normpath(os.path.join(PROJECT_ROOT, model_path))

    @property
    def trt_build_elapsed(self) -> int:
        return 0

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            self.load_error = f"Model file not found: {self.model_path}"
            print(f"[ObjectDetection] {self.load_error}")
            return

        if self._try_onnxruntime():
            return

        self._try_cv2dnn()

    def _try_onnxruntime(self) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        try:
            print("[ObjectDetection] Loading ONNX model via onnxruntime...")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            available = ort.get_available_providers()
            providers = []

            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")

            providers.append("CPUExecutionProvider")

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=providers,
            )

            inp = self.session.get_inputs()[0]
            self._input_name = inp.name
            self._output_name = self.session.get_outputs()[0].name

            if isinstance(inp.shape[2], int):
                self.img_size = inp.shape[2]

            self._backend = "ort"
            self.model_loaded = True

            print(
                f"[ObjectDetection] Model ready "
                f"(onnxruntime, provider={self.session.get_providers()[0]}, "
                f"img_size={self.img_size})."
            )

            return True

        except Exception as exc:
            print(f"[ObjectDetection] onnxruntime failed ({exc}), trying cv2.dnn...")
            return False

    def _try_cv2dnn(self):
        try:
            print("[ObjectDetection] Loading ONNX model via cv2.dnn...")

            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self._backend = "cv2dnn"
            self.model_loaded = True

            print(f"[ObjectDetection] Model ready (cv2.dnn, img_size={self.img_size}).")

        except Exception as exc:
            self.load_error = f"Failed to load ONNX model: {exc}"
            print(f"[ObjectDetection] {self.load_error}")

    def _frame_skip(self) -> int:
        try:
            return max(0, int(student.NUMBER_FRAMES_SKIPPED()))
        except Exception:
            return 0

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        return np.ascontiguousarray(img)

    def _normalize_predictions(self, raw: np.ndarray) -> np.ndarray:
        predictions = raw

        while predictions.ndim > 2:
            predictions = predictions[0]

        if predictions.ndim != 2:
            raise ValueError(f"Unexpected model output shape: {raw.shape}")

        # Handles output like [8, 25200] by transposing to [25200, 8].
        if predictions.shape[0] in (6, 7, 8) and predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T

        return predictions

    def _passes_class_specific_filter(
        self,
        bbox: Tuple[int, int, int, int],
        score: float,
        cls_id: int,
        orig_w: int,
        orig_h: int,
    ) -> bool:
        x1, y1, x2, y2 = bbox

        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)

        if box_w <= 2 or box_h <= 2:
            return False

        area = box_w * box_h
        frame_area = orig_w * orig_h
        area_ratio = area / frame_area

        width_height_ratio = box_w / box_h
        height_width_ratio = box_h / box_w

        if cls_id == 0:
            # Duckie.
            if score < self.duck_conf_threshold:
                return False

            # Ignore very high tiny yellow objects, but keep normal road-level ducks.
            if y2 < orig_h * self.duck_ignore_top_ratio:
                return False

            if box_h < orig_h * self.duck_min_height_ratio:
                return False

            if area_ratio < self.duck_min_area_ratio:
                return False

            # Reject yellow lane dashes.
            if width_height_ratio > self.duck_max_width_height_ratio:
                return False

            if height_width_ratio > self.duck_max_height_width_ratio:
                return False

            return True

        if cls_id == 1:
            # Truck.
            if score < self.truck_conf_threshold:
                return False

            if area_ratio < 0.0006:
                return False

            return True

        if cls_id == 2:
            # Sign.
            if score < self.sign_conf_threshold:
                return False

            if area_ratio < 0.0004:
                return False

            return True

        return False

    def _postprocess(self, raw: np.ndarray, orig_w: int, orig_h: int) -> List[Detection]:
        predictions = self._normalize_predictions(raw)

        n_cols = predictions.shape[1]

        if n_cols == 6:
            return self._postprocess_xyxy(predictions, orig_w, orig_h)

        if n_cols < 6:
            return []

        # YOLOv5-style output:
        # [cx, cy, w, h, object_conf, class0, class1, class2]
        if n_cols >= 8:
            obj_conf = predictions[:, 4]
            class_scores = predictions[:, 5:]

            cls_ids = np.argmax(class_scores, axis=1)
            cls_conf = class_scores[np.arange(len(class_scores)), cls_ids]
            scores = obj_conf * cls_conf

        # YOLOv8-style output:
        # [cx, cy, w, h, class0, class1, class2]
        else:
            class_scores = predictions[:, 4:]
            cls_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), cls_ids]

        mask = scores >= self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if len(predictions) == 0:
            return []

        boxes_xywh = predictions[:, :4]

        boxes_cv = [
            [int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)]
            for cx, cy, bw, bh in boxes_xywh
        ]

        indices = cv2.dnn.NMSBoxes(
            boxes_cv,
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        detections: List[Detection] = []

        for i in np.array(indices).flatten():
            cx, cy, bw, bh = boxes_xywh[i]

            bbox = _xywh2xyxy(
                cx,
                cy,
                bw,
                bh,
                self.img_size,
                orig_w,
                orig_h,
            )

            cls_id = int(cls_ids[i])
            score = float(scores[i])

            if not student.filter_by_classes(cls_id):
                continue

            if not student.filter_by_scores(score):
                continue

            if not student.filter_by_bboxes(bbox):
                continue

            if not self._passes_class_specific_filter(bbox, score, cls_id, orig_w, orig_h):
                continue

            detections.append((bbox, score, cls_id))

        return detections

    def _postprocess_xyxy(
        self,
        predictions: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        scores = predictions[:, 4]
        cls_ids = predictions[:, 5].astype(int)

        mask = scores >= self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if len(predictions) == 0:
            return []

        sx = orig_w / self.img_size
        sy = orig_h / self.img_size

        detections: List[Detection] = []

        for idx, (x1, y1, x2, y2, score, cls_id_f) in enumerate(predictions):
            bbox = (
                max(0, int(x1 * sx)),
                max(0, int(y1 * sy)),
                min(orig_w - 1, int(x2 * sx)),
                min(orig_h - 1, int(y2 * sy)),
            )

            cls_id = int(cls_ids[idx])
            score = float(scores[idx])

            if not student.filter_by_classes(cls_id):
                continue

            if not student.filter_by_scores(score):
                continue

            if not student.filter_by_bboxes(bbox):
                continue

            if not self._passes_class_specific_filter(bbox, score, cls_id, orig_w, orig_h):
                continue

            detections.append((bbox, score, cls_id))

        return detections

    def detect(self, frame_rgb: np.ndarray):
        self.frame_count += 1

        if not self.model_loaded:
            return []

        skip = self._frame_skip()

        if skip > 0 and (self.frame_count % (skip + 1)) != 0:
            return self._last_detections

        orig_h, orig_w = frame_rgb.shape[:2]

        detections = []

        try:
            # First try original frame.
            raw = self._infer(frame_rgb)
            detections = self._postprocess(raw, orig_w, orig_h)

            # If nothing is detected, try swapped color order.
            # This helps when simulation/real robot camera order differs.
            if not detections:
                swapped = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                raw_swapped = self._infer(swapped)
                detections_swapped = self._postprocess(raw_swapped, orig_w, orig_h)

                if len(detections_swapped) > len(detections):
                    detections = detections_swapped

        except Exception as exc:
            print(f"[ObjectDetection] Inference error: {exc}")
            return self._last_detections

        if self.frame_count == 1:
            print("[ObjectDetection] First frame processed")
            print(f"[ObjectDetection] backend={self._backend}, img_size={self.img_size}")

        self._last_detections = detections

        if self.frame_count % 10 == 0:
            readable = [
                (CLASS_NAMES.get(cls_id, str(cls_id)), round(score, 2), bbox)
                for bbox, score, cls_id in detections
            ]
            print(f"[ObjectDetection] detections={readable}")

        return detections

    def _infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._backend == "ort":
            inp = self._preprocess(frame_rgb)[np.newaxis]
            return self.session.run([self._output_name], {self._input_name: inp})[0]

        blob = cv2.dnn.blobFromImage(
            frame_rgb,
            1 / 255.0,
            (self.img_size, self.img_size),
            swapRB=False,
        )

        self.net.setInput(blob)
        return self.net.forward()