from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ECTVResult:
    image: np.ndarray
    iterations: int
    budget: float
    removed: float
    tv_curve: List[float]
    obj_curve: List[float]
    stop_reason: str
    p: float


@dataclass
class BudgetCertificateResult:
    image: np.ndarray
    eta: float
    budget: float
    removed: float
    p: float
    accepted_without_projection: bool


def _as_3d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x[..., None], True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"Expected 2D or 3D image, got shape {x.shape}")


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def heat_per_sample(p: float) -> float:
    if not (0.5 < p <= 1.0):
        raise ValueError(f"p must be in (0.5, 1], got {p}")
    if p == 1.0:
        return 0.0
    t = 1.0 / ((p - 0.5) ** 2) - 4.0
    return float(t * binary_entropy(p))


def delta_q(p: float, alpha: float = 1.0 / 3.0, gamma: float = 0.04) -> float:
    return float(alpha * np.tanh(gamma * heat_per_sample(p)))


def p_from_sigma(
    sigma: float,
    tau: float = 25.0 / 255.0,
    rho: float = 2.0,
    p_min: float = 0.501,
) -> float:
    sigma = max(float(sigma), 0.0)
    p = 0.5 + 0.5 * np.exp(-((sigma / tau) ** rho))
    return float(np.clip(p, p_min, 1.0))


def forward_grad(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
    gy[:-1, :, :] = u[1:, :, :] - u[:-1, :, :]
    return gx, gy


def grad_adjoint(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    out = np.zeros_like(px)
    out[:, :-1, :] -= px[:, :-1, :]
    out[:, 1:, :] += px[:, :-1, :]
    out[:-1, :, :] -= py[:-1, :, :]
    out[1:, :, :] += py[:-1, :, :]
    return out


def tv_energy(u: np.ndarray, eps: float = 1e-8, vectorial: bool = True) -> float:
    u3, _ = _as_3d(u)
    gx, gy = forward_grad(u3)
    if vectorial:
        mag = np.sqrt(np.sum(gx * gx + gy * gy, axis=2) + eps * eps)
        return float(np.mean(mag))
    mag = np.sqrt(gx * gx + gy * gy + eps * eps)
    return float(np.mean(mag))


def project_dual(
    px: np.ndarray,
    py: np.ndarray,
    radius: float,
    vectorial: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if radius <= 0:
        return np.zeros_like(px), np.zeros_like(py)

    if vectorial:
        norm = np.sqrt(np.sum(px * px + py * py, axis=2, keepdims=True))
    else:
        norm = np.sqrt(px * px + py * py)

    scale = np.maximum(1.0, norm / float(radius))
    return px / scale, py / scale


def rof_objective(u: np.ndarray, f: np.ndarray, lam: float, vectorial: bool = True) -> float:
    return float(0.5 * np.mean((u - f) ** 2) + lam * tv_energy(u, vectorial=vectorial))


def ectv_denoise(
    image: np.ndarray,
    lam: float = 0.08,
    p: Optional[float] = None,
    sigma_est: Optional[float] = None,
    tau_calib: float = 25.0 / 255.0,
    rho_calib: float = 2.0,
    alpha: float = 1.0 / 3.0,
    gamma: float = 0.04,
    max_iter: int = 120,
    primal_step: float = 0.24,
    dual_step: float = 0.24,
    theta: float = 1.0,
    tol: float = 1e-5,
    budget_enabled: bool = True,
    budget_override: Optional[float] = None,
    discrepancy_target: Optional[float] = None,
    forced_iter: Optional[int] = None,
    vectorial: bool = True,
) -> ECTVResult:
    f, squeeze = _as_3d(np.clip(image, 0.0, 1.0))
    if p is None:
        if sigma_est is None:
            raise ValueError("Provide either p or sigma_est.")
        p = p_from_sigma(sigma_est, tau=tau_calib, rho=rho_calib)

    u = f.copy()
    u_bar = u.copy()
    px = np.zeros_like(f)
    py = np.zeros_like(f)

    if budget_enabled:
        budget = float(budget_override) if budget_override is not None else delta_q(p, alpha=alpha, gamma=gamma)
    else:
        budget = np.inf
    tv0 = tv_energy(f, vectorial=vectorial)
    tv_curve = [tv0]
    obj_curve = [rof_objective(u, f, lam, vectorial=vectorial)]

    if budget_enabled and budget <= 0.0:
        out = u[..., 0] if squeeze else u
        return ECTVResult(out, 0, budget, 0.0, tv_curve, obj_curve, "zero_budget", float(p))

    stop_reason = "max_iter"
    removed = 0.0
    accepted_u = u.copy()
    accepted_removed = 0.0

    for k in range(1, max_iter + 1):
        gx, gy = forward_grad(u_bar)
        px, py = project_dual(px + dual_step * gx, py + dual_step * gy, lam, vectorial=vectorial)

        u_old = u
        kt_y = grad_adjoint(px, py)
        u = (u - primal_step * kt_y + primal_step * f) / (1.0 + primal_step)
        u = np.clip(u, 0.0, 1.0)
        u_bar = u + theta * (u - u_old)

        tv = tv_energy(u, vectorial=vectorial)
        obj = rof_objective(u, f, lam, vectorial=vectorial)
        tv_curve.append(tv)
        obj_curve.append(obj)
        removed = max(0.0, tv0 - tv)

        if forced_iter is not None and k >= forced_iter:
            stop_reason = "forced_iter"
            break

        if budget_enabled and removed > budget:
            best_u = accepted_u
            best_removed = accepted_removed
            best_tv = tv_energy(best_u, vectorial=vectorial)
            lo, hi = 0.0, 1.0
            for _ in range(24):
                eta = 0.5 * (lo + hi)
                candidate = accepted_u + eta * (u - accepted_u)
                candidate_tv = tv_energy(candidate, vectorial=vectorial)
                candidate_removed = max(0.0, tv0 - candidate_tv)
                if candidate_removed <= budget:
                    best_u = candidate.copy()
                    best_removed = candidate_removed
                    best_tv = candidate_tv
                    lo = eta
                else:
                    hi = eta
            u = best_u
            removed = best_removed
            tv_curve[-1] = best_tv
            obj_curve[-1] = rof_objective(u, f, lam, vectorial=vectorial)
            stop_reason = "budget"
            break

        accepted_u = u.copy()
        accepted_removed = removed

        if discrepancy_target is not None:
            residual_mse = float(np.mean((u - f) ** 2))
            if residual_mse >= float(discrepancy_target):
                stop_reason = "discrepancy"
                break

        rel = abs(obj_curve[-1] - obj_curve[-2]) / max(1.0, abs(obj_curve[-2]))
        if forced_iter is None and k >= 2 and rel < tol:
            stop_reason = "converged"
            break

    out = u[..., 0] if squeeze else u
    return ECTVResult(out, k, budget, removed, tv_curve, obj_curve, stop_reason, float(p))


def certify_candidate(
    image: np.ndarray,
    candidate: np.ndarray,
    p: Optional[float] = None,
    sigma_est: Optional[float] = None,
    tau_calib: float = 25.0 / 255.0,
    rho_calib: float = 2.0,
    alpha: float = 1.0 / 3.0,
    gamma: float = 0.04,
    vectorial: bool = True,
    bisection_steps: int = 24,
) -> BudgetCertificateResult:
    """Return the closest point on the noisy-to-candidate segment satisfying the TV budget."""
    f, squeeze = _as_3d(np.clip(image, 0.0, 1.0))
    v, _ = _as_3d(np.clip(candidate, 0.0, 1.0))
    if v.shape != f.shape:
        raise ValueError(f"candidate shape {v.shape} does not match image shape {f.shape}")
    if p is None:
        if sigma_est is None:
            raise ValueError("Provide either p or sigma_est.")
        p = p_from_sigma(sigma_est, tau=tau_calib, rho=rho_calib)

    budget = delta_q(p, alpha=alpha, gamma=gamma)
    tv0 = tv_energy(f, vectorial=vectorial)
    candidate_removed = max(0.0, tv0 - tv_energy(v, vectorial=vectorial))
    if candidate_removed <= budget:
        out = v[..., 0] if squeeze else v
        return BudgetCertificateResult(out, 1.0, budget, candidate_removed, float(p), True)

    best = f.copy()
    best_removed = 0.0
    lo, hi = 0.0, 1.0
    for _ in range(int(bisection_steps)):
        eta = 0.5 * (lo + hi)
        trial = f + eta * (v - f)
        trial_removed = max(0.0, tv0 - tv_energy(trial, vectorial=vectorial))
        if trial_removed <= budget:
            best = trial.copy()
            best_removed = trial_removed
            lo = eta
        else:
            hi = eta
    out = best[..., 0] if squeeze else best
    return BudgetCertificateResult(out, float(lo), budget, best_removed, float(p), False)
