from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

SPEED = 1
TURN = 0.5


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:
    left_speed = 0.0
    right_speed = 0.0

    if keys_pressed.get("up", False):
        left_speed += SPEED
        right_speed += SPEED

    if keys_pressed.get("down", False):
        left_speed -= SPEED
        right_speed -= SPEED

    if keys_pressed.get("left", False):
        left_speed -= TURN
        right_speed += TURN

    if keys_pressed.get("right", False):
        left_speed += TURN
        right_speed -= TURN

    logger.debug(f"keys={keys_pressed}, left={left_speed}, right={right_speed}")
    return left_speed, right_speed
