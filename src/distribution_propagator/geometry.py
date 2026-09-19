"""Vectorized Cartesian geometry for ensemble analysis, not propagation."""

import math
import numpy as np

TAU = 2.0 * math.pi


def cartesian_batch(elements: np.ndarray, mu: float) -> np.ndarray:
    """Convert arrays (..., 6) of elliptic equinoctial elements to SI r,v."""
    values = np.asarray(elements, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite (..., 6) orbital elements")
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be positive and finite")
    a, ex, ey, q, p, longitude = np.moveaxis(values, -1, 0)
    ecc = np.hypot(ex, ey)
    if np.any(a <= 0) or np.any(ecc >= 1) or np.any(np.hypot(q, p) > 1e6):
        raise ValueError("Unsupported elliptic orbit or retrograde singularity")
    lm = np.fmod(longitude, TAU)
    lm = np.where(lm > math.pi, lm - TAU, lm)
    lm = np.where(lm < -math.pi, lm + TAU, lm)
    lower, upper, eccentric_longitude = lm - ecc, lm + ecc, lm.copy()
    for _ in range(80):
        sine, cosine = np.sin(eccentric_longitude), np.cos(eccentric_longitude)
        residual = eccentric_longitude - ex * sine + ey * cosine - lm
        active = np.abs(residual) >= 2e-15
        if not np.any(active):
            break
        lower = np.where(active & (residual <= 0), eccentric_longitude, lower)
        upper = np.where(active & (residual > 0), eccentric_longitude, upper)
        trial = eccentric_longitude - residual / (1 - ex * cosine - ey * sine)
        trial = np.where((trial > lower) & (trial < upper), trial, 0.5 * (lower + upper))
        eccentric_longitude = np.where(active, trial, eccentric_longitude)
    else:
        raise RuntimeError("Eccentric longitude solve did not converge")
    cosine, sine = np.cos(eccentric_longitude), np.sin(eccentric_longitude)
    beta = 1 / (1 + np.sqrt(1 - ecc * ecc))
    x = a * ((1 - beta * ey * ey) * cosine + beta * ex * ey * sine - ex)
    y = a * ((1 - beta * ex * ex) * sine + beta * ex * ey * cosine - ey)
    fdot = np.sqrt(mu / a) / a / (1 - ex * cosine - ey * sine)
    vx = a * (-(1 - beta * ey * ey) * sine + beta * ex * ey * cosine) * fdot
    vy = a * ((1 - beta * ex * ex) * cosine - beta * ex * ey * sine) * fdot
    denominator = 1 + p * p + q * q
    u = np.stack(((1 - p * p + q * q) / denominator, 2 * p * q / denominator,
                  -2 * p / denominator), axis=-1)
    v = np.stack((2 * p * q / denominator, (1 + p * p - q * q) / denominator,
                  2 * q / denominator), axis=-1)
    position = x[..., None] * u + y[..., None] * v
    velocity = vx[..., None] * u + vy[..., None] * v
    return np.concatenate((position, velocity), axis=-1)


def rtn_batch(cartesian: np.ndarray) -> np.ndarray:
    """Return (..., 3, 3) matrices whose rows are the unit R,T,N vectors."""
    states = np.asarray(cartesian, dtype=float)
    if states.ndim < 1 or states.shape[-1] != 6 or not np.all(np.isfinite(states)):
        raise ValueError("Expected finite Cartesian states")
    r = states[..., :3]
    h = np.cross(r, states[..., 3:])
    rn, hn = np.linalg.norm(r, axis=-1), np.linalg.norm(h, axis=-1)
    if np.any(rn == 0) or np.any(hn == 0):
        raise ValueError("Degenerate Cartesian orbit")
    r, h = r / rn[..., None], h / hn[..., None]
    return np.stack((r, np.cross(h, r), h), axis=-2)
