from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

# Tag ID -> meaning
class TagID(IntEnum):
    STOP            = 12
    YIELD           = 11
    TURN_LEFT_FWD   = 1
    TURN_LEFT_RIGHT = 3
    TURN_RIGHT_FWD  = 4
    TURN_ALL        = 5


_TAG_TURNS = {
    TagID.TURN_LEFT_RIGHT: ["left",  "right"],
    # TagID.TURN_LEFT_FWD:   ["left", "forward"],
    TagID.TURN_LEFT_FWD:   ["left"],
    TagID.TURN_RIGHT_FWD:  ["right", "forward"],
    TagID.TURN_ALL:        ["left",  "right", "forward"],
}  # type: Dict[int, List[str]]


# State enum
class State(IntEnum):
    MOVING      = auto()
    SLOWING     = auto()
    YIELD_CREEP = auto()
    STOPPED     = auto()
    CHECKPATH   = auto()
    POST_STOP   = auto()
    INTERSECT   = auto()
    PRE_TURN    = auto()
    TURNING     = auto()
    EXITING     = auto()


# Config
@dataclass
class SignBehaviorConfig:
    # Red-line detection
    red_strip_frac  = 0.50
    red_pixel_frac  = 0.03
    red_hsv_low1    = (0,   120,  80)   # type: Tuple[int, int, int]
    red_hsv_high1   = (10,  255, 255)   # type: Tuple[int, int, int]
    red_hsv_low2    = (170, 120,  80)   # type: Tuple[int, int, int]
    red_hsv_high2   = (180, 255, 255)   # type: Tuple[int, int, int]

    # Pre-turn alignment
    preturn_align_tolerance = 0.03
    preturn_align_speed     = 0.001

    # Stop-line hold
    stop_hold_frames = 0

    # Speed ramp
    slow_ramp_factor        = 0.92
    stopped_speed_threshold = 0.05

    # YIELD specific
    yield_min_speed    = 0.1
    yield_creep_frames = 5

    # CHECKPATH sweep
    check_left_frames   = 4
    check_right_frames  = 4
    check_turn_speed    = 0.04
    check_settle_frames = 5

    # POST_STOP
    post_stop_frames = 15 # prev 12
    post_stop_speed  = 0.2

    # Pre-turn forward creep 
    preturn_right_frames = 9  #prev 30
    preturn_left_frames  = 20  # prev 40
    preturn_speed        = 0.20

    # Intersection manoeuvres
    intersect_forward_frames = 25   # sachiroa tore lane servoing marjvniv gaushvebs 
    intersect_left_frames    = 18   # prev 9
    intersect_right_frames   = 18   # prev 9
    intersect_forward_speed  = (0.20, 0.20)  # type: Tuple[float, float]
    
    # changed so one v isn't 0 - no turning in place
    intersect_left_speed     = (0.15, 0.22)  # type: Tuple[float, float]
    intersect_right_speed    = (0.22, 0.15)  # type: Tuple[float, float]

    # Exiting
    exit_speed            = 0.20
    exit_right_min_frac   = 0.02
    exit_timeout_frames   = 10
    exit_post_line_frames = 5
    exit_frames_forward   = 8
    exit_frames_left      = 0
    exit_frames_right     = 0

    # Vehicle detection
    vehicle_class_id      = 1
    vehicle_min_bbox_area = 2000.0

    # AprilTag detector
    min_margin         = 10.0
    tag_confirm_frames = 3
    camera_params      = None   # type: Optional[Tuple[float, float, float, float]]
    tag_size_m         = 0.065