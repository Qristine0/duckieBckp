from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, List, Tuple


class TagID(IntEnum):
    TURN_RIGHT_FWD = 9
    TURN_LEFT_FWD = 10
    TURN_LEFT_RIGHT = 11
    STOP = 26
    YIELD = 39


# Maps raw detected AprilTag IDs to logical sign meanings.
_TAG_ID_MAP: Dict[int, TagID] = {
    # Your terminal shows STOP sign is detected as raw tag 1 in simulation.
    1: TagID.STOP,

    9: TagID.TURN_RIGHT_FWD,
    10: TagID.TURN_LEFT_FWD,
    11: TagID.TURN_LEFT_RIGHT,

    20: TagID.STOP,
    24: TagID.STOP,
    25: TagID.STOP,
    26: TagID.STOP,

    39: TagID.YIELD,
}


def resolve_tag(raw_id: int):
    """Map raw detected tag ID to logical TagID, or None if unknown."""
    return _TAG_ID_MAP.get(raw_id)


_TAG_TURNS = {
    TagID.TURN_LEFT_RIGHT: ["left", "right"],
    TagID.TURN_LEFT_FWD: ["left", "forward"],
    TagID.TURN_RIGHT_FWD: ["right", "forward"],
}  # type: Dict[int, List[str]]


class State(IntEnum):
    MOVING = auto()
    SLOWING = auto()
    YIELD_CREEP = auto()
    STOPPED = auto()
    CHECKPATH = auto()
    POST_STOP = auto()
    INTERSECT = auto()
    PRE_TURN = auto()
    TURNING = auto()
    EXITING = auto()


@dataclass
class SignBehaviorConfig:
    # Red-line detection
    red_strip_frac = 0.32
    red_pixel_frac = 0.025

    # Bigger = stop later / closer to red line.
    # Smaller = stop earlier / farther from red line.
    red_line_close_y2_ratio = 0.82

    red_hsv_low1 = (0, 120, 80)       # type: Tuple[int, int, int]
    red_hsv_high1 = (10, 255, 255)    # type: Tuple[int, int, int]
    red_hsv_low2 = (170, 120, 80)     # type: Tuple[int, int, int]
    red_hsv_high2 = (180, 255, 255)   # type: Tuple[int, int, int]

    # After sign is detected, keep driving but slower while approaching red line.
    approach_speed_factor = 0.72
    approach_ramp_factor = 0.98

    # Stop-line hold
    stop_hold_frames = 0

    # Braking after close red line is detected
    slow_ramp_factor = 0.84
    stopped_speed_threshold = 0.06

    # YIELD specific
    yield_min_speed = 0.1
    yield_creep_frames = 5

    # CHECKPATH sweep
    check_left_frames = 4
    check_right_frames = 4
    check_turn_speed = 0.04
    check_settle_frames = 5

    # POST_STOP
    post_stop_frames = 25
    post_stop_speed = 0.2

    # Pre-turn forward creep
    preturn_right_frames = 11
    preturn_left_frames = 20
    preturn_speed = 0.20

    # Intersection manoeuvres
    intersect_forward_frames = 28
    intersect_left_frames = 18
    intersect_right_frames = 18
    intersect_forward_speed = (0.20, 0.20)  # type: Tuple[float, float]

    # No turning in place
    intersect_left_speed = (0.15, 0.22)     # type: Tuple[float, float]
    intersect_right_speed = (0.22, 0.15)    # type: Tuple[float, float]

    # Exiting
    exit_speed = 0.20
    exit_timeout_frames = 5

