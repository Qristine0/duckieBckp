from typing import Tuple
import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, int]:
    """
    Convert encoder ticks to wheel rotation (radians)
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    dphi = 2 * np.pi * (ticks - prev_ticks) / resolution
    return dphi, ticks


def wrap_angle(theta: float) -> float:
    """
    Normalize angle to [-pi, pi]
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:
    """
    Differential drive odometry using midpoint method
    """

    if baseline <= 0:
        raise ValueError("baseline must be > 0")

    # Wheel distances
    d_left = R * delta_phi_left
    d_right = R * delta_phi_right

    # Linear displacement
    d_A = (d_right + d_left) / 2.0

    # Angular displacement
    delta_theta = (d_right - d_left) / baseline

    # Midpoint heading (IMPORTANT)
    theta_mid = theta_prev + delta_theta / 2.0

    # Position update
    x_curr = x_prev + d_A * np.cos(theta_mid)
    y_curr = y_prev + d_A * np.sin(theta_mid)

    # Orientation update
    theta_curr = wrap_angle(theta_prev + delta_theta)

    return x_curr, y_curr , theta_curr