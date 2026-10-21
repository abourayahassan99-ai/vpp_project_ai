"""
Temporal smoothing of tracked quad corners (Agent 3).

Model: every corner coordinate is an independent constant-velocity Kalman filter
(state = [position, velocity]).  Measurements are weighted by tracker confidence so a shaky
frame barely moves the estimate while a confident one is followed closely.  A Rauch-Tung-Striebel
backward pass then removes the lag a forward-only filter would have (the pipeline is offline),
and fills short gaps by pure prediction.

Smoothing never crosses a LOST gap: callers pass one contiguous segment at a time.
"""

from __future__ import annotations

import numpy as np


def smooth_segment(
    quads: np.ndarray,
    confidence: np.ndarray,
    process_noise: float = 4.0,
    measurement_noise: float = 1.0,
    speed_ref: float = 3.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    quads : (N, 4, 2) float array. Rows where ``confidence <= 0`` (or non-finite) are treated as
        missing: the filter predicts through them.
    confidence : (N,) in [0, 1].
    process_noise : acceleration variance (px^2, working scale) at rest; scaled up with corner speed (see speed_ref).
    speed_ref : corner speed (px/frame) at which the process noise has doubled.
    measurement_noise : measurement variance (px^2) at confidence 1.

    Returns (N, 4, 2) smoothed quads.
    """
    quads = np.asarray(quads, np.float64)
    n = len(quads)
    if n < 3:
        return quads.astype(np.float32)

    z = quads.reshape(n, 8)  # 8 independent channels
    conf = np.asarray(confidence, np.float64)
    # Adaptive process noise: stiff smoothing while the corners are nearly still, loose during fast zoom / pan so the
    # filter does not lag (a fixed noise level left the ad edge 10-18 px off the TV during fast zooms).
    speed = np.zeros(n)
    if n > 2:
        sp = np.abs(np.diff(z, axis=0)).mean(axis=1)
        speed[1:] = sp
        speed[0] = sp[0]
        speed = np.convolve(np.pad(speed, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
    qscale = 1.0 + (speed / max(speed_ref, 1e-3)) ** 2
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = process_noise * np.array([[0.25, 0.5], [0.5, 1.0]])
    I2 = np.eye(2)

    xs = np.zeros((n, 8, 2))   # filtered state
    Ps = np.zeros((n, 2, 2))   # filtered covariance (same for all channels -> shared)
    xp = np.zeros((n, 8, 2))   # predicted state
    Pp = np.zeros((n, 2, 2))   # predicted covariance

    # initialise from first measured sample
    first = int(np.argmax(conf > 0))
    x = np.zeros((8, 2))
    x[:, 0] = z[first]
    P = np.diag([measurement_noise, 25.0])

    for t in range(n):
        if t == 0:
            xpred, Ppred = x, P
        else:
            xpred = x @ F.T
            Ppred = F @ P @ F.T + Q * qscale[t]
        xp[t], Pp[t] = xpred, Ppred

        if conf[t] > 1e-3 and np.isfinite(z[t]).all():
            R = measurement_noise / max(conf[t] ** 2, 0.05)
            S = Ppred[0, 0] + R
            K = Ppred[:, 0] / S                      # (2,)
            innov = z[t] - xpred[:, 0]               # (8,)
            x = xpred + innov[:, None] * K[None, :]
            P = (I2 - np.outer(K, [1.0, 0.0])) @ Ppred
        else:
            x, P = xpred, Ppred
        xs[t], Ps[t] = x, P

    # RTS backward pass
    sm = xs.copy()
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        sm[t] = xs[t] + (sm[t + 1] - xp[t + 1]) @ C.T
    return sm[:, :, 0].reshape(n, 4, 2).astype(np.float32)
