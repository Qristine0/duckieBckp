import sys
import os
import signal
import threading
import time
import socket

script_dir = os.path.dirname(os.path.abspath(_file_))
project_root = os.path.normpath(os.path.join(script_dir, "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import cv2
from flask import Flask, Response, render_template_string, jsonify, request

from tasks.sign_detection.packages.agent_with_signs import LaneServoingAgentWithSigns

from tasks.sign_detection.packages.detection import (
    detect_obstacles,
    CLASS_NAMES,
    reset_detection_state,
)
from tasks.sign_detection.packages.detection import should_stop as student_should_stop

from servers.templates.sign_detection import SIGN_DETECTION_TEMPLATE as HTML_TEMPLATE

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs


app = Flask(_name_)

lane_agent = None
camera = None
wheels = None

running = False
manual_mode = False
stop_event = threading.Event()

_last_detections = []
_detection_lock = threading.Lock()

_stopped_by_det = False
_stop_reason = ""

keys_pressed = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
}

_keys_lock = threading.Lock()
_keys_last_update = time.time()


def manual_control_loop():
    global _keys_last_update

    while not stop_event.is_set():
        if not manual_mode or wheels is None:
            time.sleep(0.05)
            continue

        if time.time() - _keys_last_update > 0.5:
            with _keys_lock:
                for key in keys_pressed:
                    keys_pressed[key] = False

        with _keys_lock:
            key_state = keys_pressed.copy()

        left = 0.0
        right = 0.0

        if key_state["up"]:
            left, right = 0.5, 0.5

        if key_state["down"]:
            left, right = -0.5, -0.5

        if key_state["up"] and key_state["left"]:
            left, right = 0.2, 0.5
        elif key_state["up"] and key_state["right"]:
            left, right = 0.5, 0.2
        elif key_state["left"]:
            left, right = -0.3, 0.3
        elif key_state["right"]:
            left, right = 0.3, -0.3

        wheels.set_wheels_speed(left, right)
        time.sleep(0.05)


def _should_stop(detections):
    # Your detection.py currently returns False.
    # Keep this wrapper so duck/truck stop can be re-enabled later.
    return student_should_stop(detections)


def _draw_detections(frame_bgr, detections):
    for bbox, score, cls_id in detections:
        x1, y1, x2, y2 = bbox

        if cls_id == 0:
            color = (0, 215, 255)
        elif cls_id == 1:
            color = (180, 100, 220)
        else:
            color = (50, 205, 50)

        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def visualize(frame_bgr):
    global _stopped_by_det, _stop_reason, _last_detections

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detections = detect_obstacles(frame_rgb)

    with _detection_lock:
        _last_detections = detections

    should_stop_flag, reason = _should_stop(detections)

    _stopped_by_det = bool(should_stop_flag)
    _stop_reason = reason

    _draw_detections(frame_bgr, detections)

    if manual_mode:
        return frame_bgr

    if wheels is None:
        return frame_bgr

    if not running:
        wheels.set_wheels_speed(0.0, 0.0)
        return frame_bgr

    if lane_agent is None:
        wheels.set_wheels_speed(0.0, 0.0)
        return frame_bgr

    if should_stop_flag:
        wheels.set_wheels_speed(0.0, 0.0)
        return frame_bgr

    # Important:
    # Pass detections into LaneServoingAgentWithSigns.
    left, right = lane_agent.compute_commands(frame_rgb, detections)
    wheels.set_wheels_speed(left, right)

    return frame_bgr


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50, rgb=False)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, hostname=socket.gethostname(), virtual=False)


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def reset_runtime_state():
    global _last_detections, _stopped_by_det, _stop_reason

    with _detection_lock:
        _last_detections = []

    _stopped_by_det = False
    _stop_reason = ""

    reset_detection_state()

    if lane_agent is not None and hasattr(lane_agent, "reset_behavior"):
        lane_agent.reset_behavior()

    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)

    print("[Server] runtime state reset")


@app.route("/start", methods=["POST"])
def start():
    global running

    reset_runtime_state()
    running = True

    return jsonify({"status": "running"})


@app.route("/stop", methods=["POST"])
def stop():
    global running

    running = False

    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)

    return jsonify({"status": "stopped"})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    global manual_mode

    mode = request.json.get("mode", "auto") if request.json else "auto"
    manual_mode = mode == "manual"

    if wheels is not None and not manual_mode:
        wheels.set_wheels_speed(0.0, 0.0)

    return jsonify({"mode": "manual" if manual_mode else "auto"})


@app.route("/keys", methods=["POST"])
def update_keys():
    global _keys_last_update

    data = request.json or {}

    with _keys_lock:
        for key in keys_pressed:
            keys_pressed[key] = bool(data.get(key, False))

    _keys_last_update = time.time()

    return jsonify({"status": "ok"})


@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    return jsonify({"conf_threshold": 0.5})


@app.route("/status")
def status():
    with _detection_lock:
        detections = list(_last_detections)

    sign_state = None
    sign_debug = {}

    if lane_agent is not None:
        sign_state = getattr(lane_agent, "sign_state", None)
        sign_debug = getattr(lane_agent, "sign_debug", {}) or {}

    return jsonify({
        "running": running,
        "manual_mode": manual_mode,
        "model_loaded": True,
        "load_error": None,
        "trt_building": False,
        "stopped_by_detection": _stopped_by_det,
        "stop_reason": _stop_reason,
        "conf_threshold": 0.5,
        "sign_state": sign_state,
        "sign_debug": sign_debug,
        "detections": [
            {
                "class": CLASS_NAMES.get(cls_id, str(cls_id)),
                "score": round(score, 3),
                "bbox": list(bbox),
            }
            for bbox, score, cls_id in detections
        ],
    })


def main():
    global lane_agent, camera, wheels

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    suppress_http_logs()

    print("=" * 60)
    print("SIGN DETECTION — LANE FOLLOW + SIGNS + RED LINE")
    print("=" * 60)

    def _init_wheels():
        global wheels

        wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
        print("[Init] Wheels ready")

    def _init_camera():
        global camera

        cam = CameraDriver()
        cam.start()
        camera = cam
        print("[Init] Camera ready")

    def _init_agent():
        global lane_agent

        lane_agent = LaneServoingAgentWithSigns()
        print(f"[Init] Lane agent with signs ready (speed={lane_agent.base_speed})")

    threading.Thread(target=_init_wheels, daemon=True).start()
    threading.Thread(target=_init_camera, daemon=True).start()
    threading.Thread(target=_init_agent, daemon=True).start()
    threading.Thread(target=manual_control_loop, daemon=True).start()

    def _shutdown(signum, frame):
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    web_port = find_available_port(args.port)

    print(f"\nWeb Interface: http://{socket.gethostname()}.local:{web_port}")
    print("=" * 60 + "\n")

    try:
        app.run(host="0.0.0.0", port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if _name_ == "_main_":
    sys.exit(main())