from typing import Tuple
import os
import yaml
import numpy as np


# -------- Load config --------
GAINS_FILE = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..',
    'config',
    'modcon_config.yaml'
)

try:
    with open(GAINS_FILE, "r") as f:
        _g = yaml.safe_load(f) or {}
except FileNotFoundError:
    _g = {}

# Gains (safe defaults)
K_P = float(_g.get('k_P', 2.5))
K_I = float(_g.get('k_I', 0.0))
K_D = float(_g.get('k_D', 0.3))

MAX_OMEGA = float(_g.get('max_omega', 8.0))
MIN_OMEGA = -MAX_OMEGA

MAX_INT = float(_g.get('max_integral', 1.5))


# -------- Helpers --------
def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_error(theta_ref: float, theta_hat: float) -> float:
    """
    Shortest angular difference
    """
    return wrap_angle(theta_ref - theta_hat)


# -------- PID --------
def PIDController(
    v_0: float,
    theta_ref: float,
    theta_hat: float,
    prev_e: float,
    prev_int: float,
    delta_t: float,
) -> Tuple[float, float, float, float]:
    """
    Stable PID controller (no internal stopping logic)
    """

    # Stable timestep
    delta_t = max(delta_t, 0.01)

    # Wrapped error
    e = angle_error(theta_ref, theta_hat)

    # Integral (anti-windup)
    integral = prev_int + delta_t * e
    integral = np.clip(integral, -MAX_INT, MAX_INT)

    # Derivative
    deriv = (e - prev_e) / delta_t

    # PID output
    u = K_P * e + K_I * integral + K_D * deriv
    u = float(np.clip(u, MIN_OMEGA, MAX_OMEGA))

    return v_0, u, e, integral