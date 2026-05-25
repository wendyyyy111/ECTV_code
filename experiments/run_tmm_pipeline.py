from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from skimage import color
from skimage.feature import canny
from skimage.filters import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import binary_dilation, disk
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_tv_chambolle, denoise_wavelet, estimate_sigma
from skimage.transform import resize

try:
    import bm3d
except Exception:  # pragma: no cover - optional baseline
    bm3d = None

try:
    import h5py
except Exception:  # pragma: no cover - optional learned baseline reader
    h5py = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ectv import certify_candidate, delta_q, ectv_denoise, heat_per_sample, p_from_sigma, tv_energy

FIG3 = ROOT / "figure3"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures_tmm"
GENERATED = ROOT / "generated"

METHOD_ORDER = [
    "noisy",
    "tv_maxiter",
    "tv_fixedk_val",
    "tv_discrepancy",
    "tv_chambolle_val",
    "ectv",
    "ectv_fidelity",
    "gaussian_val",
    "median_val",
    "bilateral_val",
    "pm_diff_val",
    "wavelet_bayes",
    "nlm",
    "bm3d",
    "fdncnn_color",
    "ectv_fdncnn",
]
DOWNSTREAM_METHOD_ORDER = [
    "noisy",
    "tv_maxiter",
    "tv_fixedk_val",
    "tv_discrepancy",
    "tv_chambolle_val",
    "ectv",
    "ectv_fidelity",
    "bilateral_val",
    "pm_diff_val",
    "wavelet_bayes",
    "nlm",
    "bm3d",
    "fdncnn_color",
    "ectv_fdncnn",
]
METHOD_LABELS = {
    "noisy": "Noisy",
    "tv_maxiter": "TV-NoBudget",
    "tv_fixedk_val": "TV-FixedK-Val",
    "tv_discrepancy": "TV-Discrepancy",
    "tv_chambolle_val": "TV-Chambolle-Val",
    "ectv": "ECTV-Tight",
    "ectv_fidelity": "ECTV-Fidelity",
    "gaussian_val": "Gaussian-Val",
    "median_val": "Median-Val",
    "bilateral_val": "Bilateral-Val",
    "pm_diff_val": "PM-Diff-Val",
    "wavelet_bayes": "Wavelet-Bayes",
    "nlm": "NLM",
    "bm3d": "BM3D-RGB",
    "fdncnn_color": "FDnCNN-Color",
    "ectv_fdncnn": "ECTV-Cert-FDnCNN",
    "ectv_budget": "ECTV-budget",
    "tv_fixed20": "TV-Fixed20",
    "tv_oracle": "TV-Oracle",
}

BASELINE_DESCRIPTIONS = {
    "noisy": ("Input", "No denoising"),
    "tv_maxiter": ("ECTV solver", "TV evolution with budget stopping disabled"),
    "tv_fixedk_val": ("ECTV solver", "Fixed iteration count selected on calibration images"),
    "tv_discrepancy": ("ECTV solver", "Stop when residual MSE reaches the estimated noise variance"),
    "tv_chambolle_val": ("scikit-image", "ROF/TV-Chambolle weight selected on calibration images"),
    "ectv": ("Proposed", "Tight removed-TV budget preset with conservative sensitivity"),
    "ectv_fidelity": ("Proposed", "Fidelity-oriented budget preset with calibration-selected lambda/alpha by noise regime"),
    "gaussian_val": ("OpenCV", "Gaussian blur sigma selected on calibration images"),
    "median_val": ("OpenCV", "Median kernel selected on calibration images"),
    "bilateral_val": ("scikit-image", "Bilateral parameters selected on calibration images"),
    "pm_diff_val": ("NumPy/OpenCV", "Perona-Malik iterations and conductance selected on calibration images"),
    "wavelet_bayes": ("scikit-image", "BayesShrink wavelet denoising"),
    "nlm": ("scikit-image", "Non-local means with estimated sigma"),
    "bm3d": ("bm3d package", "Color BM3D when RGB input is provided"),
    "fdncnn_color": ("Official DnCNN/FDnCNN weights", "Color CNN denoiser with noise-level input"),
    "ectv_fdncnn": ("Proposed certificate wrapper", "FDnCNN-Color proposal accepted or projected to the ECTV-Fidelity removed-TV budget"),
}

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{idx:02d}.png"
KODAK_COUNT = 24
FDNCNN_COLOR_MODEL = DATA / "models" / "FDnCNN_color.mat"
_FDNCNN_COLOR_CACHE = None


def ensure_dirs() -> None:
    for path in [
        DATA / "raw" / "kodak24",
        DATA / "processed" / "kodak24",
        DATA / "processed" / "polyu_local",
        RESULTS,
        FIGS,
        GENERATED,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path, max_side: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        arr = resize(arr, (int(round(h * scale)), int(round(w * scale))), anti_aliasing=True)
        arr = np.asarray(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def save_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr8 = np.clip(np.asarray(arr) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def download_kodak() -> List[Path]:
    raw = DATA / "raw" / "kodak24"
    processed = DATA / "processed" / "kodak24"
    paths: List[Path] = []
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
    openers = [urllib.request.build_opener()]
    if proxy:
        openers.insert(0, urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy})))
    else:
        openers.append(urllib.request.build_opener(urllib.request.ProxyHandler({"http": "http://127.0.0.1:33210", "https": "http://127.0.0.1:33210"})))

    for idx in range(1, KODAK_COUNT + 1):
        raw_path = raw / f"kodim{idx:02d}.png"
        proc_path = processed / f"kodim{idx:02d}.png"
        if proc_path.exists():
            paths.append(proc_path)
            continue
        if not raw_path.exists():
            last_exc = None
            for opener in openers:
                try:
                    with opener.open(KODAK_URL.format(idx=idx), timeout=60) as resp:
                        raw_path.write_bytes(resp.read())
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                print(f"[warn] Kodak download failed for {idx:02d}: {last_exc}")
                continue
        if not proc_path.exists():
            arr = read_image(raw_path, max_side=256)
            save_image(proc_path, arr)
        paths.append(proc_path)
    return paths


def prepare_local_polyu() -> List[Tuple[str, Path, Path]]:
    def is_nontrivial_pair(noisy_path: Path, gt_path: Path) -> bool:
        try:
            noisy_arr = read_image(noisy_path, max_side=256)
            gt_arr = read_image(gt_path, max_side=256)
        except Exception:
            return False
        return float(np.mean((noisy_arr - gt_arr) ** 2)) > 1e-12

    processed_full = DATA / "processed" / "polyu_full"
    if processed_full.exists():
        pairs = []
        for noisy in sorted(processed_full.glob("*_noisy.png")):
            gt = noisy.with_name(noisy.name.replace("_noisy.png", "_gt.png"))
            if not gt.exists():
                continue
            if not is_nontrivial_pair(noisy, gt):
                continue
            stem = noisy.stem.replace("_noisy", "")
            pairs.append((stem, noisy, gt))
        if pairs:
            return pairs

    full_root = DATA / "raw" / "PolyU-Real-World-Noisy-Images-Dataset-master"
    if full_root.exists():
        out = DATA / "processed" / "polyu_full"
        out.mkdir(parents=True, exist_ok=True)
        pairs = []
        for noisy_src in sorted(full_root.rglob("*_real.JPG")):
            gt_src = noisy_src.with_name(noisy_src.name.replace("_real.JPG", "_mean.JPG"))
            if not gt_src.exists():
                continue
            stem = noisy_src.stem.replace("_real", "")
            noisy = out / f"{stem}_noisy.png"
            gt = out / f"{stem}_gt.png"
            if not noisy.exists():
                save_image(noisy, read_image(noisy_src, max_side=256))
            if not gt.exists():
                save_image(gt, read_image(gt_src, max_side=256))
            if not is_nontrivial_pair(noisy, gt):
                continue
            pairs.append((stem, noisy, gt))
        if pairs:
            return pairs

    pairs = []
    names = ["bicycle", "door", "plant", "stair", "toy", "waterhouse"]
    out = DATA / "processed" / "polyu_local"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        noisy_src = FIG3 / f"{name}.JPG"
        gt_src = FIG3 / f"{name}mean.JPG"
        if not noisy_src.exists() or not gt_src.exists():
            continue
        noisy = out / f"{name}_noisy.png"
        gt = out / f"{name}_gt.png"
        save_image(noisy, read_image(noisy_src, max_side=256))
        save_image(gt, read_image(gt_src, max_side=256))
        pairs.append((name, noisy, gt))
    return pairs


def legacy_images() -> List[Path]:
    paths = []
    for name in ["Lena.jpg", "Baboon.jpg", "Barbara.jpg", "Peppers.jpg"]:
        path = FIG3 / name
        if path.exists():
            paths.append(path)
    return paths


def to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return color.rgb2gray(arr)


def add_gaussian(clean: np.ndarray, sigma255: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, sigma255 / 255.0, clean.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method).replace("_", r"\_")


def translate_image(clean: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = clean.shape[:2]
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    out = cv2.warpAffine(
        clean.astype(np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def default_baseline_config() -> Dict[str, float | int | str]:
    return {
        "tv_fixedk": 20,
        "tv_chambolle_scale": 1.0,
        "gaussian_sigma": 1.0,
        "median_kernel": 3,
        "bilateral_color_scale": 1.0,
        "bilateral_spatial": 4,
        "pm_iterations": 10,
        "pm_kappa_scale": 2.0,
        "calibration_source": "default",
        "ectv_gamma_low": 0.04,
        "ectv_gamma_high": 0.32,
        "ectv_fidelity_low_lam": 0.08,
        "ectv_fidelity_low_alpha": 1.0 / 3.0,
        "ectv_fidelity_mid_lam": 0.12,
        "ectv_fidelity_mid_alpha": 1.0 / 3.0,
        "ectv_fidelity_high_lam": 0.18,
        "ectv_fidelity_high_alpha": 0.75,
    }


def ectv_fidelity_params_from_sigma(sigma_est: float, config: Dict[str, float | int | str] | None = None) -> Tuple[float, float, float]:
    cfg = default_baseline_config()
    if config:
        cfg.update(config)
    sigma255 = 255.0 * float(max(sigma_est, 0.0))
    if sigma255 < 20.0:
        key = "low"
    elif sigma255 < 40.0:
        key = "mid"
    else:
        key = "high"
    lam = float(cfg[f"ectv_fidelity_{key}_lam"])
    alpha = float(cfg[f"ectv_fidelity_{key}_alpha"])
    gamma = float(cfg["ectv_gamma_high"])
    return lam, alpha, gamma


class FDnCNNColor:
    def __init__(self, model_path: Path):
        if h5py is None:
            raise RuntimeError("h5py is required to load FDnCNN_color.mat")
        self.layers = []
        self._file = h5py.File(model_path, "r")

        def decode(ds) -> str:
            arr = np.asarray(ds).squeeze()
            return "".join(chr(int(x)) for x in arr.flatten())

        for ref in self._file["net/layers"][:, 0]:
            group = self._file[ref]
            layer_type = decode(group["type"])
            if layer_type == "conv":
                weight_refs = np.asarray(group["weights"]).flatten()
                weights = np.asarray(self._file[weight_refs[0]], dtype=np.float32)
                bias = np.asarray(self._file[weight_refs[1]], dtype=np.float32).reshape(-1)
                self.layers.append(("conv", weights, bias))
            elif layer_type == "relu":
                self.layers.append(("relu", None, None))
            elif layer_type == "concat":
                continue

    @staticmethod
    def _conv_same(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
        windows = sliding_window_view(padded, (3, 3), axis=(0, 1))
        columns = windows.reshape(x.shape[0] * x.shape[1], -1)
        y = columns @ weights.reshape(weights.shape[0], -1).T + bias.reshape(1, -1)
        return y.reshape(x.shape[0], x.shape[1], weights.shape[0]).astype(np.float32)

    def denoise(self, image: np.ndarray, sigma_est: float) -> np.ndarray:
        sigma_map = np.full((*image.shape[:2], 1), float(max(sigma_est, 0.0)), dtype=np.float32)
        x = np.concatenate([image.astype(np.float32), sigma_map], axis=2)
        for layer_type, weights, bias in self.layers:
            if layer_type == "conv":
                x = self._conv_same(x, weights, bias)
            else:
                x = np.maximum(x, 0.0)
        return np.clip(x, 0.0, 1.0).astype(np.float32)


def get_fdncnn_color() -> FDnCNNColor | None:
    global _FDNCNN_COLOR_CACHE
    if h5py is None or not FDNCNN_COLOR_MODEL.exists():
        return None
    if _FDNCNN_COLOR_CACHE is None:
        _FDNCNN_COLOR_CACHE = FDnCNNColor(FDNCNN_COLOR_MODEL)
    return _FDNCNN_COLOR_CACHE


def perona_malik_diffusion(
    image: np.ndarray,
    sigma_est: float,
    iterations: int,
    kappa_scale: float,
    step: float = 0.16,
) -> np.ndarray:
    u = np.asarray(image, dtype=np.float32).copy()
    kappa = max(float(kappa_scale) * max(float(sigma_est), 1e-6), 1e-3)
    for _ in range(int(iterations)):
        north = np.zeros_like(u)
        south = np.zeros_like(u)
        east = np.zeros_like(u)
        west = np.zeros_like(u)
        north[1:, :, :] = u[:-1, :, :] - u[1:, :, :]
        south[:-1, :, :] = u[1:, :, :] - u[:-1, :, :]
        west[:, 1:, :] = u[:, :-1, :] - u[:, 1:, :]
        east[:, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
        update = (
            np.exp(-(north / kappa) ** 2) * north
            + np.exp(-(south / kappa) ** 2) * south
            + np.exp(-(east / kappa) ** 2) * east
            + np.exp(-(west / kappa) ** 2) * west
        )
        u = np.clip(u + step * update, 0.0, 1.0)
    return u.astype(np.float32)


def calibrate_baselines(calib_images: List[Tuple[str, str, Path]], quick: bool) -> Dict[str, float | int | str]:
    """Select simple baseline parameters on calibration images only."""
    config = default_baseline_config()
    if not calib_images:
        return config

    examples = calib_images[:1] if quick else calib_images[:4]
    sigma = 25.0 / 255.0
    cases = []
    for dataset, name, path in examples:
        clean = read_image(path, max_side=256)
        noisy = add_gaussian(clean, 25, 0)
        cases.append((dataset, name, clean, noisy))
    if not cases:
        return config

    def score(outputs: Iterable[np.ndarray]) -> float:
        vals = []
        for (dataset, name, clean, noisy), out in zip(cases, outputs):
            vals.append(metric_row(clean, noisy, out)["psnr"])
        return float(np.mean(vals))

    best_k = int(config["tv_fixedk"])
    best_score = -np.inf
    for k in [5, 10, 20, 40, 80]:
        outs = [ectv_denoise(noisy, sigma_est=sigma, max_iter=80, budget_enabled=False, forced_iter=k).image for _, _, clean, noisy in cases]
        val = score(outs)
        if val > best_score:
            best_score, best_k = val, k
    config["tv_fixedk"] = best_k

    best_scale = float(config["tv_chambolle_scale"])
    best_score = -np.inf
    for scale in [0.25, 0.5, 1.0, 1.5, 2.0]:
        outs = [
            np.clip(denoise_tv_chambolle(noisy, weight=scale * sigma, channel_axis=-1).astype(np.float32), 0.0, 1.0)
            for _, _, clean, noisy in cases
        ]
        val = score(outs)
        if val > best_score:
            best_score, best_scale = val, scale
    config["tv_chambolle_scale"] = best_scale

    best_g = float(config["gaussian_sigma"])
    best_score = -np.inf
    for sigma_space in [0.6, 1.0, 1.4]:
        outs = [np.clip(cv2.GaussianBlur(noisy, (0, 0), sigmaX=sigma_space, sigmaY=sigma_space), 0.0, 1.0) for _, _, clean, noisy in cases]
        val = score(outs)
        if val > best_score:
            best_score, best_g = val, sigma_space
    config["gaussian_sigma"] = best_g

    best_kernel = int(config["median_kernel"])
    best_score = -np.inf
    for kernel in [3, 5]:
        outs = [np.clip(cv2.medianBlur(noisy.astype(np.float32), kernel), 0.0, 1.0) for _, _, clean, noisy in cases]
        val = score(outs)
        if val > best_score:
            best_score, best_kernel = val, kernel
    config["median_kernel"] = best_kernel

    best_bilateral = (float(config["bilateral_color_scale"]), int(config["bilateral_spatial"]))
    best_score = -np.inf
    for color_scale in [0.75, 1.0, 1.5]:
        for sigma_spatial in [3, 5]:
            outs = [
                np.clip(
                    denoise_bilateral(
                        noisy,
                        sigma_color=max(color_scale * sigma, 1e-6),
                        sigma_spatial=sigma_spatial,
                        channel_axis=-1,
                    ).astype(np.float32),
                    0.0,
                    1.0,
                )
                for _, _, clean, noisy in cases
            ]
            val = score(outs)
            if val > best_score:
                best_score, best_bilateral = val, (color_scale, sigma_spatial)
    config["bilateral_color_scale"], config["bilateral_spatial"] = best_bilateral

    best_pm = (int(config["pm_iterations"]), float(config["pm_kappa_scale"]))
    best_score = -np.inf
    for iterations in [5, 10, 20, 40]:
        for kappa_scale in [1.0, 2.0, 4.0]:
            outs = [perona_malik_diffusion(noisy, sigma, iterations=iterations, kappa_scale=kappa_scale) for _, _, clean, noisy in cases]
            val = score(outs)
            if val > best_score:
                best_score, best_pm = val, (iterations, kappa_scale)
    config["pm_iterations"], config["pm_kappa_scale"] = best_pm

    fidelity_candidates = [
        (0.08, 1.0 / 3.0),
        (0.12, 1.0 / 3.0),
        (0.18, 0.50),
        (0.18, 0.75),
        (0.25, 0.75),
    ]
    fidelity_bins = [("low", 15.0), ("mid", 25.0), ("high", 50.0)]
    for bin_name, sigma255 in fidelity_bins:
        sigma_bin = sigma255 / 255.0
        bin_cases = []
        for dataset, name, path in examples:
            clean = read_image(path, max_side=256)
            noisy = add_gaussian(clean, sigma255, int(round(sigma255)))
            bin_cases.append((dataset, name, clean, noisy))
        best_pair = (
            float(config[f"ectv_fidelity_{bin_name}_lam"]),
            float(config[f"ectv_fidelity_{bin_name}_alpha"]),
        )
        best_score = -np.inf
        for lam, alpha in fidelity_candidates:
            vals = []
            for _, _, clean, noisy in bin_cases:
                out = ectv_denoise(
                    noisy,
                    sigma_est=sigma_bin,
                    lam=lam,
                    alpha=alpha,
                    gamma=float(config["ectv_gamma_high"]),
                    max_iter=80,
                    budget_enabled=True,
                    tol=0.0,
                ).image
                vals.append(metric_row(clean, noisy, out)["psnr"])
            val = float(np.mean(vals))
            if val > best_score:
                best_score, best_pair = val, (lam, alpha)
        config[f"ectv_fidelity_{bin_name}_lam"] = best_pair[0]
        config[f"ectv_fidelity_{bin_name}_alpha"] = best_pair[1]

    config["calibration_source"] = ",".join(name for _, name, _, _ in cases)
    return config


def metric_row(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray) -> Dict[str, float]:
    ch_axis = -1 if clean.ndim == 3 else None
    data_range = 1.0
    min_side = min(clean.shape[:2])
    win_size = 7 if min_side >= 7 else max(3, min_side // 2 * 2 - 1)
    psnr = peak_signal_noise_ratio(clean, out, data_range=data_range)
    ssim = structural_similarity(clean, out, channel_axis=ch_axis, data_range=data_range, win_size=win_size)

    gc = sobel(to_gray(clean))
    go = sobel(to_gray(out))
    denom = float(np.linalg.norm(gc.ravel()) * np.linalg.norm(go.ravel()) + 1e-12)
    grad_corr = float(np.dot(gc.ravel(), go.ravel()) / denom)
    grad_mae = float(np.mean(np.abs(gc - go)))
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "grad_corr": grad_corr,
        "grad_mae": grad_mae,
        "tv_input": tv_energy(noisy),
        "tv_output": tv_energy(out),
    }


def run_methods(
    noisy: np.ndarray,
    sigma_est: float,
    max_iter: int,
    baseline_config: Dict[str, float | int | str] | None = None,
    include_learned: bool = True,
) -> Dict[str, Tuple[np.ndarray, Dict[str, float], object]]:
    cfg = default_baseline_config()
    if baseline_config:
        cfg.update(baseline_config)
    outputs: Dict[str, Tuple[np.ndarray, Dict[str, float], object]] = {}
    outputs["noisy"] = (noisy, {"iterations": 0, "runtime": 0.0, "budget": 0.0, "removed": 0.0, "p": 0.0}, None)

    t0 = time.perf_counter()
    ectv_gamma = float(cfg["ectv_gamma_low"])
    ectv = ectv_denoise(noisy, sigma_est=sigma_est, max_iter=max_iter, budget_enabled=True, gamma=ectv_gamma)
    outputs["ectv"] = (
        ectv.image,
        {
            "iterations": ectv.iterations,
            "runtime": time.perf_counter() - t0,
            "budget": ectv.budget,
            "removed": ectv.removed,
            "p": ectv.p,
            "gamma": ectv_gamma,
        },
        ectv,
    )

    t0 = time.perf_counter()
    fidelity_lam, fidelity_alpha, fidelity_gamma = ectv_fidelity_params_from_sigma(sigma_est, cfg)
    ectv_fidelity = ectv_denoise(
        noisy,
        sigma_est=sigma_est,
        lam=fidelity_lam,
        alpha=fidelity_alpha,
        gamma=fidelity_gamma,
        max_iter=max_iter,
        budget_enabled=True,
        tol=0.0,
    )
    outputs["ectv_fidelity"] = (
        ectv_fidelity.image,
        {
            "iterations": ectv_fidelity.iterations,
            "runtime": time.perf_counter() - t0,
            "budget": ectv_fidelity.budget,
            "removed": ectv_fidelity.removed,
            "p": ectv_fidelity.p,
            "lambda": fidelity_lam,
            "alpha": fidelity_alpha,
            "gamma": fidelity_gamma,
        },
        ectv_fidelity,
    )

    t0 = time.perf_counter()
    tv_maxiter = ectv_denoise(noisy, sigma_est=sigma_est, max_iter=max_iter, budget_enabled=False)
    outputs["tv_maxiter"] = (
        tv_maxiter.image,
        {
            "iterations": tv_maxiter.iterations,
            "runtime": time.perf_counter() - t0,
            "budget": float("inf"),
            "removed": tv_maxiter.removed,
            "p": tv_maxiter.p,
        },
        tv_maxiter,
    )

    t0 = time.perf_counter()
    tv_fixedk = ectv_denoise(
        noisy,
        sigma_est=sigma_est,
        max_iter=max_iter,
        budget_enabled=False,
        forced_iter=int(cfg["tv_fixedk"]),
    )
    outputs["tv_fixedk_val"] = (
        tv_fixedk.image,
        {
            "iterations": tv_fixedk.iterations,
            "runtime": time.perf_counter() - t0,
            "budget": float("nan"),
            "removed": tv_fixedk.removed,
            "p": tv_fixedk.p,
        },
        tv_fixedk,
    )

    t0 = time.perf_counter()
    tv_discrepancy = ectv_denoise(
        noisy,
        sigma_est=sigma_est,
        max_iter=max_iter,
        budget_enabled=False,
        discrepancy_target=max(float(sigma_est) ** 2, 1e-10),
    )
    outputs["tv_discrepancy"] = (
        tv_discrepancy.image,
        {
            "iterations": tv_discrepancy.iterations,
            "runtime": time.perf_counter() - t0,
            "budget": float("nan"),
            "removed": tv_discrepancy.removed,
            "p": tv_discrepancy.p,
        },
        tv_discrepancy,
    )

    t0 = time.perf_counter()
    try:
        tv_ch = denoise_tv_chambolle(noisy, weight=float(cfg["tv_chambolle_scale"]) * max(float(sigma_est), 1e-6), channel_axis=-1)
        outputs["tv_chambolle_val"] = (
            np.clip(tv_ch.astype(np.float32), 0.0, 1.0),
            {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] TV-Chambolle failed: {exc}")

    t0 = time.perf_counter()
    try:
        sigma_space = float(cfg["gaussian_sigma"])
        gauss = cv2.GaussianBlur(noisy.astype(np.float32), (0, 0), sigmaX=sigma_space, sigmaY=sigma_space)
        outputs["gaussian_val"] = (
            np.clip(gauss.astype(np.float32), 0.0, 1.0),
            {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] Gaussian failed: {exc}")

    t0 = time.perf_counter()
    try:
        median = cv2.medianBlur(noisy.astype(np.float32), int(cfg["median_kernel"]))
        outputs["median_val"] = (
            np.clip(median.astype(np.float32), 0.0, 1.0),
            {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] Median failed: {exc}")

    t0 = time.perf_counter()
    try:
        bilateral = denoise_bilateral(
            noisy,
            sigma_color=max(float(cfg["bilateral_color_scale"]) * max(float(sigma_est), 1e-6), 1e-6),
            sigma_spatial=int(cfg["bilateral_spatial"]),
            channel_axis=-1,
        )
        outputs["bilateral_val"] = (
            np.clip(bilateral.astype(np.float32), 0.0, 1.0),
            {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] Bilateral failed: {exc}")

    t0 = time.perf_counter()
    try:
        pm = perona_malik_diffusion(
            noisy,
            sigma_est=max(float(sigma_est), 1e-6),
            iterations=int(cfg["pm_iterations"]),
            kappa_scale=float(cfg["pm_kappa_scale"]),
        )
        outputs["pm_diff_val"] = (
            np.clip(pm.astype(np.float32), 0.0, 1.0),
            {"iterations": int(cfg["pm_iterations"]), "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] Perona-Malik diffusion failed: {exc}")

    t0 = time.perf_counter()
    try:
        wavelet = denoise_wavelet(noisy, sigma=max(float(sigma_est), 1e-6), method="BayesShrink", mode="soft", channel_axis=-1, rescale_sigma=True)
        outputs["wavelet_bayes"] = (
            np.clip(wavelet.astype(np.float32), 0.0, 1.0),
            {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] Wavelet failed: {exc}")

    t0 = time.perf_counter()
    try:
        nlm_sigma = float(np.mean(estimate_sigma(noisy, channel_axis=-1)))
        nlm = denoise_nl_means(noisy, h=0.85 * nlm_sigma, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=-1)
        outputs["nlm"] = (
            np.clip(nlm.astype(np.float32), 0.0, 1.0),
            {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
            None,
        )
    except Exception as exc:
        print(f"[warn] NLM failed: {exc}")

    if bm3d is not None:
        t0 = time.perf_counter()
        try:
            if noisy.ndim == 3 and noisy.shape[2] == 3 and hasattr(bm3d, "bm3d_rgb"):
                bm3d_out = bm3d.bm3d_rgb(noisy.astype(np.float32), float(max(sigma_est, 1e-6)), profile="np")
            else:
                bm3d_out = bm3d.bm3d(noisy.astype(np.float32), float(max(sigma_est, 1e-6)), profile="np")
            outputs["bm3d"] = (
                np.clip(np.asarray(bm3d_out, dtype=np.float32), 0.0, 1.0),
                {"iterations": 0, "runtime": time.perf_counter() - t0, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
                None,
            )
        except Exception as exc:
            print(f"[warn] BM3D failed: {exc}")

    if include_learned:
        fdncnn = get_fdncnn_color()
        if fdncnn is not None:
            t0 = time.perf_counter()
            try:
                fdncnn_out = fdncnn.denoise(noisy, sigma_est=max(float(sigma_est), 1e-6))
                fdncnn_runtime = time.perf_counter() - t0
                outputs["fdncnn_color"] = (
                    fdncnn_out,
                    {"iterations": 0, "runtime": fdncnn_runtime, "budget": float("nan"), "removed": float("nan"), "p": 0.0},
                    None,
                )
                cert_t0 = time.perf_counter()
                cert = certify_candidate(
                    noisy,
                    fdncnn_out,
                    sigma_est=max(float(sigma_est), 1e-6),
                    alpha=fidelity_alpha,
                    gamma=fidelity_gamma,
                )
                outputs["ectv_fdncnn"] = (
                    cert.image,
                    {
                        "iterations": 0,
                        "runtime": fdncnn_runtime + (time.perf_counter() - cert_t0),
                        "budget": cert.budget,
                        "removed": cert.removed,
                        "p": cert.p,
                        "eta": cert.eta,
                        "accepted": 1.0 if cert.accepted_without_projection else 0.0,
                        "alpha": fidelity_alpha,
                        "gamma": fidelity_gamma,
                    },
                    cert,
                )
            except Exception as exc:
                print(f"[warn] FDnCNN-Color failed: {exc}")

    return outputs


def run_synthetic(
    images: List[Tuple[str, str, Path]],
    max_iter: int,
    quick: bool,
    baseline_config: Dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    sigmas = [15, 25] if quick else [15, 25, 50]
    seeds = [0] if quick else [0, 1, 2]
    example_saved = False
    energy_saved = False

    for dataset, name, path in images:
        clean = read_image(path, max_side=256)
        for sigma in sigmas:
            for seed in seeds:
                noisy = add_gaussian(clean, sigma, seed)
                outputs = run_methods(noisy, sigma / 255.0, max_iter=max_iter, baseline_config=baseline_config)
                for method, (out, meta, result) in outputs.items():
                    row = {
                        "task": "synthetic_image",
                        "dataset": dataset,
                        "image": name,
                        "sigma255": sigma,
                        "seed": seed,
                        "method": method,
                    }
                    row.update(metric_row(clean, noisy, out))
                    row.update(meta)
                    rows.append(row)
                if not example_saved and dataset == "kodak24":
                    make_synthetic_panel(clean, noisy, outputs, sigma, seed)
                    example_saved = True
                if not energy_saved:
                    make_energy_panel(outputs["ectv"][2], outputs["tv_maxiter"][2], name)
                    energy_saved = True
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "synthetic_image.csv", index=False)
    return df


def run_real_noise(pairs: List[Tuple[str, Path, Path]], max_iter: int, baseline_config: Dict[str, float | int | str]) -> pd.DataFrame:
    rows = []
    example_saved = False
    for name, noisy_path, gt_path in pairs:
        noisy = read_image(noisy_path, max_side=256)
        clean = read_image(gt_path, max_side=256)
        sigma_est = float(np.mean(estimate_sigma(noisy, channel_axis=-1)))
        outputs = run_methods(noisy, sigma_est, max_iter=max_iter, baseline_config=baseline_config)
        for method, (out, meta, _) in outputs.items():
            dataset_name = "polyu_full" if "polyu_full" in noisy_path.parts else "polyu_local"
            row = {
                "task": "real_camera",
                "dataset": dataset_name,
                "image": name,
                "sigma255": sigma_est * 255.0,
                "seed": -1,
                "method": method,
            }
            row.update(metric_row(clean, noisy, out))
            row.update(meta)
            rows.append(row)
        if not example_saved:
            make_real_panel(clean, noisy, outputs, name)
            example_saved = True
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "real_camera.csv", index=False)
    return df


def make_video_frames(clean: np.ndarray, n: int = 8) -> List[np.ndarray]:
    frames = []
    for i in range(n):
        dy = float(i - n // 2)
        dx = float((i % 4) - 2)
        frames.append(translate_image(clean, dx=dx, dy=dy))
    return frames


def run_video(images: List[Tuple[str, str, Path]], max_iter: int, baseline_config: Dict[str, float | int | str]) -> pd.DataFrame:
    dataset, name, path = images[0]
    clean_frames = make_video_frames(read_image(path, max_side=256), n=8)
    sigma = 20
    noisy_frames = [add_gaussian(f, sigma, seed=i) for i, f in enumerate(clean_frames)]
    rows = []
    method_outputs: Dict[str, List[np.ndarray]] = {"noisy": noisy_frames}

    for idx, noisy in enumerate(noisy_frames):
        outputs = run_methods(noisy, sigma / 255.0, max_iter=max_iter, baseline_config=baseline_config)
        for method, (out, meta, _) in outputs.items():
            if method not in method_outputs:
                method_outputs[method] = []
            if method != "noisy":
                method_outputs[method].append(out)
            row = {
                "task": "video_sequence",
                "dataset": f"{dataset}_synthetic_sequence",
                "image": f"{name}_frame_{idx:02d}",
                "sigma255": sigma,
                "seed": idx,
                "method": method,
            }
            row.update(metric_row(clean_frames[idx], noisy, out))
            row.update(meta)
            rows.append(row)

    for method, outs in method_outputs.items():
        trf = temporal_residual_fluctuation(clean_frames, outs)
        for row in rows:
            if row["method"] == method:
                row["temporal_residual_fluctuation"] = trf
    make_video_panel(clean_frames, noisy_frames, method_outputs, name)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "video_sequence.csv", index=False)
    return df


def temporal_residual_fluctuation(clean: List[np.ndarray], out: List[np.ndarray]) -> float:
    vals = []
    for t in range(1, min(len(clean), len(out))):
        vals.append(np.mean(np.abs((out[t] - out[t - 1]) - (clean[t] - clean[t - 1]))))
    return float(np.mean(vals)) if vals else float("nan")


def jpeg_roundtrip(arr: np.ndarray, quality: int) -> Tuple[np.ndarray, float]:
    im = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    bpp = 8.0 * len(buf.getvalue()) / (arr.shape[0] * arr.shape[1])
    buf.seek(0)
    rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return rec, float(bpp)


def run_compression(
    images: List[Tuple[str, str, Path]],
    max_iter: int,
    quick: bool,
    baseline_config: Dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    qualities = [30, 50] if quick else [10, 20, 30, 50, 70]
    examples = images[:2] if quick else images[:4]
    for dataset, name, path in examples:
        clean = read_image(path, max_side=256)
        noisy = add_gaussian(clean, 25, 0)
        outputs = run_methods(noisy, 25 / 255.0, max_iter=max_iter, baseline_config=baseline_config, include_learned=True)
        for method in DOWNSTREAM_METHOD_ORDER:
            if method not in outputs:
                continue
            out = outputs[method][0]
            for quality in qualities:
                rec, bpp = jpeg_roundtrip(out, quality)
                row = {
                    "task": "jpeg_compression",
                    "dataset": dataset,
                    "image": name,
                    "quality": quality,
                    "method": method,
                    "bpp": bpp,
                }
                row.update(metric_row(clean, noisy, rec))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "downstream_compression.csv", index=False)
    make_compression_panel(df)
    return df


def run_feature_matching(
    images: List[Tuple[str, str, Path]],
    max_iter: int,
    quick: bool,
    baseline_config: Dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    examples = images[:2] if quick else images[:5]
    sigmas = [25, 50]
    detectors = ["orb"]
    if hasattr(cv2, "SIFT_create"):
        detectors.append("sift")
    for dataset, name, path in examples:
        clean1 = read_image(path, max_side=256)
        dx, dy = 9.0, 6.0
        clean2 = translate_image(clean1, dx=dx, dy=dy)
        gt_h = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)
        for sigma in sigmas:
            noisy1 = add_gaussian(clean1, sigma, 1 + sigma)
            noisy2 = add_gaussian(clean2, sigma, 2 + sigma)
            outs1 = run_methods(noisy1, sigma / 255.0, max_iter=max_iter, baseline_config=baseline_config, include_learned=True)
            outs2 = run_methods(noisy2, sigma / 255.0, max_iter=max_iter, baseline_config=baseline_config, include_learned=True)
            for method in DOWNSTREAM_METHOD_ORDER:
                if method not in outs1 or method not in outs2:
                    continue
                for detector in detectors:
                    metrics, drawn = feature_match_metrics(outs1[method][0], outs2[method][0], detector=detector, gt_h=gt_h)
                    row = {"task": "feature_matching", "dataset": dataset, "image": name, "sigma255": sigma, "method": method, "detector": detector}
                    row.update(metrics)
                    rows.append(row)
                    if sigma == 25 and detector == "orb" and name == examples[0][1] and method in ["noisy", "tv_maxiter", "ectv", "ectv_fidelity"] and drawn is not None:
                        save_image(FIGS / f"match_{method}.png", drawn[..., ::-1] / 255.0)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "downstream_feature_matching.csv", index=False)
    make_matching_panel(df)
    return df


def edge_prf(
    clean_img: np.ndarray,
    restored_img: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float = 0.10,
    high_threshold: float = 0.20,
    tolerance_px: int = 1,
) -> Dict[str, float]:
    clean = to_gray(clean_img).astype(np.float32)
    restored = to_gray(restored_img).astype(np.float32)
    e_ref = canny(clean, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    e_pred = canny(restored, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

    selem = disk(tolerance_px)
    ref_dil = binary_dilation(e_ref, selem)
    pred_dil = binary_dilation(e_pred, selem)

    pred_count = int(e_pred.sum())
    ref_count = int(e_ref.sum())
    precision_hits = int(np.logical_and(e_pred, ref_dil).sum())
    recall_hits = int(np.logical_and(e_ref, pred_dil).sum())

    precision = precision_hits / pred_count if pred_count else 0.0
    recall = recall_hits / ref_count if ref_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "edge_density": float(pred_count / e_pred.size),
        "pred_edge_count": float(pred_count),
        "ref_edge_count": float(ref_count),
        "canny_sigma": float(sigma),
        "canny_low": float(low_threshold),
        "canny_high": float(high_threshold),
        "tolerance_px": float(tolerance_px),
    }


def edge_agreement_map(
    clean_img: np.ndarray,
    restored_img: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float = 0.10,
    high_threshold: float = 0.20,
    tolerance_px: int = 1,
) -> np.ndarray:
    clean = to_gray(clean_img).astype(np.float32)
    restored = to_gray(restored_img).astype(np.float32)
    e_ref = canny(clean, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    e_pred = canny(restored, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    selem = disk(tolerance_px)
    ref_dil = binary_dilation(e_ref, selem)
    pred_dil = binary_dilation(e_pred, selem)
    matched = np.logical_and(e_pred, ref_dil)
    false_pred = np.logical_and(e_pred, np.logical_not(ref_dil))
    missed = np.logical_and(e_ref, np.logical_not(pred_dil))
    rgb = np.ones((*e_ref.shape, 3), dtype=np.float32)
    rgb[matched] = np.array([0.0, 0.65, 0.0], dtype=np.float32)
    rgb[false_pred] = np.array([0.85, 0.0, 0.0], dtype=np.float32)
    rgb[missed] = np.array([0.0, 0.25, 0.95], dtype=np.float32)
    return rgb


def run_edge_extraction(
    images: List[Tuple[str, str, Path]],
    max_iter: int,
    quick: bool,
    baseline_config: Dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    examples = images[:2] if quick else images[:5]
    sigmas = [25, 50]
    example_payload = None
    for dataset, name, path in examples:
        clean = read_image(path, max_side=256)
        for sigma in sigmas:
            noisy = add_gaussian(clean, sigma, 3 + sigma)
            outputs = run_methods(noisy, sigma / 255.0, max_iter=max_iter, baseline_config=baseline_config, include_learned=True)
            for method in DOWNSTREAM_METHOD_ORDER:
                if method not in outputs:
                    continue
                metrics = edge_prf(clean, outputs[method][0], tolerance_px=1)
                metrics_tol2 = edge_prf(clean, outputs[method][0], tolerance_px=2)
                metrics["edge_f1_tol2"] = metrics_tol2["edge_f1"]
                row = {"task": "edge_extraction", "dataset": dataset, "image": name, "sigma255": sigma, "method": method}
                row.update(metrics)
                rows.append(row)
            if example_payload is None and sigma == 25:
                example_payload = (clean, outputs, name)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "downstream_edge.csv", index=False)
    if example_payload is not None:
        make_edge_panel(*example_payload)
    return df


def make_tracking_sequence(clean: np.ndarray, n: int = 8, dx: float = 2.0, dy: float = 1.0) -> List[np.ndarray]:
    return [translate_image(clean, dx=i * dx, dy=i * dy) for i in range(n)]


def _gray_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(to_gray(arr) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def klt_metrics_for_sequence(clean_frames: List[np.ndarray], outputs: List[np.ndarray], dx: float, dy: float) -> List[Dict[str, float]]:
    first_gray = _gray_u8(clean_frames[0])
    pts = cv2.goodFeaturesToTrack(first_gray, maxCorners=300, qualityLevel=0.01, minDistance=5, blockSize=5)
    if pts is None or len(pts) == 0:
        return []
    base_pts = pts.reshape(-1, 2).astype(np.float32)
    h, w = first_gray.shape
    rows = []
    for t in range(len(outputs) - 1):
        prev = _gray_u8(outputs[t])
        nxt = _gray_u8(outputs[t + 1])
        p0_xy = base_pts + np.array([t * dx, t * dy], dtype=np.float32)
        margin = 8
        mask = (
            (p0_xy[:, 0] >= margin)
            & (p0_xy[:, 0] < w - margin)
            & (p0_xy[:, 1] >= margin)
            & (p0_xy[:, 1] < h - margin)
        )
        p0_xy = p0_xy[mask]
        if len(p0_xy) == 0:
            continue
        p0 = p0_xy.reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, p0, None, winSize=(21, 21), maxLevel=3)
        if p1 is None or st is None:
            rows.append({"transition": float(t), "klt_survival": 0.0, "klt_epe": float("nan"), "klt_bad1": float("nan"), "klt_bad3": float("nan")})
            continue
        valid = st.reshape(-1).astype(bool)
        survival = float(valid.mean()) if len(valid) else 0.0
        if valid.any():
            flow = p1.reshape(-1, 2)[valid] - p0_xy[valid]
            err = flow - np.array([dx, dy], dtype=np.float32)
            epe = np.sqrt(np.sum(err * err, axis=1))
            rows.append(
                {
                    "transition": float(t),
                    "klt_survival": survival,
                    "klt_epe": float(np.mean(epe)),
                    "klt_bad1": float(np.mean(epe > 1.0)),
                    "klt_bad3": float(np.mean(epe > 3.0)),
                }
            )
        else:
            rows.append({"transition": float(t), "klt_survival": survival, "klt_epe": float("nan"), "klt_bad1": float("nan"), "klt_bad3": float("nan")})
    return rows


def farneback_metrics_for_sequence(outputs: List[np.ndarray], dx: float, dy: float) -> List[Dict[str, float]]:
    rows = []
    for t in range(len(outputs) - 1):
        prev = _gray_u8(outputs[t])
        nxt = _gray_u8(outputs[t + 1])
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        margin = 12
        valid_flow = flow[margin:-margin, margin:-margin, :] if min(flow.shape[:2]) > 2 * margin else flow
        err = valid_flow - np.array([dx, dy], dtype=np.float32)
        epe = np.sqrt(np.sum(err * err, axis=2))
        rows.append(
            {
                "transition": float(t),
                "farneback_epe": float(np.mean(epe)),
                "farneback_median_epe": float(np.median(epe)),
                "farneback_bad1": float(np.mean(epe > 1.0)),
                "farneback_bad3": float(np.mean(epe > 3.0)),
                "farneback_bias_dx": float(np.mean(valid_flow[..., 0] - dx)),
                "farneback_bias_dy": float(np.mean(valid_flow[..., 1] - dy)),
            }
        )
    return rows


def run_tracking(
    images: List[Tuple[str, str, Path]],
    max_iter: int,
    quick: bool,
    baseline_config: Dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    examples = images[:1] if quick else images[:5]
    sigmas = [20, 50]
    dx, dy = 2.0, 1.0
    for dataset, name, path in examples:
        clean_frames = make_tracking_sequence(read_image(path, max_side=256), n=8, dx=dx, dy=dy)
        for sigma in sigmas:
            noisy_frames = [add_gaussian(f, sigma, seed=100 + sigma + i) for i, f in enumerate(clean_frames)]
            method_outputs: Dict[str, List[np.ndarray]] = {}
            for noisy in noisy_frames:
                outputs = run_methods(noisy, sigma / 255.0, max_iter=max_iter, baseline_config=baseline_config, include_learned=True)
                for method, (out, _, _) in outputs.items():
                    method_outputs.setdefault(method, []).append(out)
            for method in DOWNSTREAM_METHOD_ORDER:
                if method not in method_outputs:
                    continue
                klt_rows = klt_metrics_for_sequence(clean_frames, method_outputs[method], dx=dx, dy=dy)
                fb_rows = farneback_metrics_for_sequence(method_outputs[method], dx=dx, dy=dy)
                by_transition = {int(r["transition"]): r for r in klt_rows}
                for fb in fb_rows:
                    t = int(fb["transition"])
                    row = {
                        "task": "tracking_flow",
                        "dataset": dataset,
                        "image": name,
                        "sigma255": sigma,
                        "method": method,
                        "transition": t,
                        "dx": dx,
                        "dy": dy,
                    }
                    row.update(by_transition.get(t, {}))
                    row.update(fb)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "downstream_tracking.csv", index=False)
    make_tracking_panel(df)
    return df


def delta_q_linear_clip(p: float, alpha: float = 1.0 / 3.0, gamma: float = 0.04) -> float:
    return float(np.clip(alpha * gamma * heat_per_sample(p), 0.0, alpha))


def ablation_row(
    clean: np.ndarray,
    noisy: np.ndarray,
    out: np.ndarray,
    result,
    dataset: str,
    name: str,
    ablation: str,
    method: str,
    p: float,
    lam: float,
    gamma: float,
    mapping: str,
    runtime: float,
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {
        "task": "ablation",
        "dataset": dataset,
        "image": name,
        "sigma255": 25,
        "seed": 0,
        "ablation": ablation,
        "method": method,
        "p": float(p),
        "lambda": float(lam),
        "gamma": float(gamma),
        "mapping": mapping,
        "iterations": float(result.iterations),
        "runtime": float(runtime),
        "budget": float(result.budget),
        "removed": float(result.removed),
    }
    row.update(metric_row(clean, noisy, out))
    return row


def ablation_metric_row(
    clean: np.ndarray,
    noisy: np.ndarray,
    out: np.ndarray,
    dataset: str,
    name: str,
    ablation: str,
    method: str,
    p: float,
    lam: float,
    gamma: float,
    mapping: str,
    iterations: float,
    runtime: float,
    budget: float,
    removed: float,
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {
        "task": "ablation",
        "dataset": dataset,
        "image": name,
        "sigma255": 25,
        "seed": 0,
        "ablation": ablation,
        "method": method,
        "p": float(p),
        "lambda": float(lam),
        "gamma": float(gamma),
        "mapping": mapping,
        "iterations": float(iterations),
        "runtime": float(runtime),
        "budget": float(budget),
        "removed": float(removed),
    }
    row.update(metric_row(clean, noisy, out))
    return row


def run_ablation(
    images: List[Tuple[str, str, Path]],
    max_iter: int,
    quick: bool,
    baseline_config: Dict[str, float | int | str],
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    examples = images[:1] if quick else images[:3]
    p_values = [0.55, 0.65, 0.75, 0.85, 0.95]
    lam_values = [0.04, 0.08, 0.12, 0.16]
    gamma_values = [0.02, 0.04, 0.08]
    sigma_est = 25.0 / 255.0
    p_auto = p_from_sigma(sigma_est)
    cfg = default_baseline_config()
    cfg.update(baseline_config)
    gamma_default = float(cfg["ectv_gamma_low"])

    for dataset, name, path in examples:
        clean = read_image(path, max_side=256)
        noisy = add_gaussian(clean, 25, 0)

        for p_value in p_values:
            t0 = time.perf_counter()
            result = ectv_denoise(noisy, p=p_value, lam=0.08, gamma=gamma_default, max_iter=max_iter, budget_enabled=True)
            rows.append(
                ablation_row(
                    clean,
                    noisy,
                    result.image,
                    result,
                    dataset,
                    name,
                    "p_sweep",
                    "ectv",
                    p_value,
                    0.08,
                    gamma_default,
                    "tanh",
                    time.perf_counter() - t0,
                )
            )

        for lam in lam_values:
            t0 = time.perf_counter()
            result = ectv_denoise(noisy, p=p_auto, lam=lam, gamma=gamma_default, max_iter=max_iter, budget_enabled=True)
            rows.append(
                ablation_row(
                    clean,
                    noisy,
                    result.image,
                    result,
                    dataset,
                    name,
                    "lambda_sweep",
                    "ectv",
                    p_auto,
                    lam,
                    gamma_default,
                    "tanh",
                    time.perf_counter() - t0,
                )
            )

        for mapping in ["tanh", "linear_clip"]:
            for gamma in gamma_values:
                budget_override = delta_q_linear_clip(p_auto, gamma=gamma) if mapping == "linear_clip" else None
                t0 = time.perf_counter()
                result = ectv_denoise(
                    noisy,
                    p=p_auto,
                    lam=0.08,
                    gamma=gamma,
                    max_iter=max_iter,
                    budget_enabled=True,
                    budget_override=budget_override,
                )
                rows.append(
                    ablation_row(
                        clean,
                        noisy,
                        result.image,
                        result,
                        dataset,
                        name,
                        "mapping_gamma",
                        "ectv",
                        p_auto,
                        0.08,
                        gamma,
                        mapping,
                        time.perf_counter() - t0,
                    )
                )

        stopping_runs = [
            ("ectv_budget", {"budget_enabled": True}),
            ("tv_fixedk_val", {"budget_enabled": False, "forced_iter": int(baseline_config["tv_fixedk"])}),
            ("tv_discrepancy", {"budget_enabled": False, "discrepancy_target": sigma_est**2}),
            ("tv_maxiter", {"budget_enabled": False}),
        ]
        for method, kwargs in stopping_runs:
            t0 = time.perf_counter()
            result = ectv_denoise(noisy, p=p_auto, lam=0.08, gamma=gamma_default, max_iter=max_iter, **kwargs)
            rows.append(
                ablation_row(
                    clean,
                    noisy,
                    result.image,
                    result,
                    dataset,
                    name,
                    "stopping",
                    method,
                    p_auto,
                    0.08,
                    gamma_default,
                    "tanh",
                    time.perf_counter() - t0,
                )
            )

        t0 = time.perf_counter()
        tv_ch = denoise_tv_chambolle(
            noisy,
            weight=float(baseline_config["tv_chambolle_scale"]) * sigma_est,
            channel_axis=-1,
        )
        tv_ch = np.clip(tv_ch.astype(np.float32), 0.0, 1.0)
        rows.append(
            ablation_metric_row(
                clean,
                noisy,
                tv_ch,
                dataset,
                name,
                "stopping",
                "tv_chambolle_val",
                p_auto,
                0.08,
                gamma_default,
                "validation",
                0,
                time.perf_counter() - t0,
                float("nan"),
                max(0.0, tv_energy(noisy) - tv_energy(tv_ch)),
            )
        )

        best_oracle = None
        oracle_t0 = time.perf_counter()
        for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80]:
            if k > max_iter:
                continue
            result = ectv_denoise(noisy, p=p_auto, lam=0.08, gamma=gamma_default, max_iter=max_iter, budget_enabled=False, forced_iter=k)
            psnr = metric_row(clean, noisy, result.image)["psnr"]
            if best_oracle is None or psnr > best_oracle[0]:
                best_oracle = (psnr, result)
        if best_oracle is not None:
            oracle_runtime = time.perf_counter() - oracle_t0
            result = best_oracle[1]
            rows.append(
                ablation_row(
                    clean,
                    noisy,
                    result.image,
                    result,
                    dataset,
                    name,
                    "stopping",
                    "tv_oracle",
                    p_auto,
                    0.08,
                    gamma_default,
                    "clean_reference",
                    oracle_runtime,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "ablation.csv", index=False)
    make_ablation_panel(df)
    return df


def feature_match_metrics(
    a: np.ndarray,
    b: np.ndarray,
    detector: str,
    gt_h: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray | None]:
    gray1 = (to_gray(a) * 255).astype(np.uint8)
    gray2 = (to_gray(b) * 255).astype(np.uint8)
    if detector == "sift" and hasattr(cv2, "SIFT_create"):
        feature = cv2.SIFT_create(nfeatures=700)
        norm = cv2.NORM_L2
    else:
        feature = cv2.ORB_create(nfeatures=700)
        norm = cv2.NORM_HAMMING
        detector = "orb"
    kp1, des1 = feature.detectAndCompute(gray1, None)
    kp2, des2 = feature.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            "keypoints_1": len(kp1),
            "keypoints_2": len(kp2),
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "corner_error": float("nan"),
            "homography_success": 0.0,
        }, None
    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:150]
    inliers = 0
    corner_error = float("nan")
    success = 0.0
    if len(matches) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        h_est, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.sum())
        if h_est is not None:
            h, w = gray1.shape
            corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
            pred = cv2.perspectiveTransform(corners, h_est)
            ref = cv2.perspectiveTransform(corners, gt_h)
            err = np.sqrt(np.sum((pred.reshape(-1, 2) - ref.reshape(-1, 2)) ** 2, axis=1))
            corner_error = float(np.mean(err))
            success = 1.0 if corner_error < 3.0 and inliers >= 10 else 0.0
    drawn = cv2.drawMatches(
        (np.clip(a * 255, 0, 255)).astype(np.uint8),
        kp1,
        (np.clip(b * 255, 0, 255)).astype(np.uint8),
        kp2,
        matches[:30],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return {
        "keypoints_1": len(kp1),
        "keypoints_2": len(kp2),
        "matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(1, len(matches))),
        "corner_error": corner_error,
        "homography_success": success,
    }, drawn


def abs_error(clean: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(clean - out), axis=2)


def grad_diff(clean: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.abs(sobel(to_gray(clean)) - sobel(to_gray(out)))


def add_image(ax, arr: np.ndarray, title: str, cmap=None) -> None:
    ax.imshow(np.clip(arr, 0, 1), cmap=cmap)
    ax.set_title(title, fontsize=7, pad=1.0)
    ax.axis("off")


def make_budget_curves() -> None:
    ps = np.linspace(0.501, 0.999, 400)
    entropy = np.array([-p * np.log(p) - (1 - p) * np.log(1 - p) for p in ps])
    temp = np.array([1.0 / ((p - 0.5) ** 2) - 4.0 for p in ps])
    q = np.array([heat_per_sample(float(p)) for p in ps])
    dq = np.array([delta_q(float(p)) for p in ps])
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5))
    for ax, y, title in zip(axs, [temp, entropy, q, dq], ["T(p)", "H_b(p)", "q(p)", "Delta_Q(p)"]):
        ax.plot(ps, y, color="#1f77b4", lw=1.8)
        ax.set_xlabel("p")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def make_synthetic_panel(clean, noisy, outputs, sigma, seed) -> None:
    methods = [m for m in ["noisy", "tv_maxiter", "tv_chambolle_val", "ectv", "ectv_fidelity", "wavelet_bayes", "nlm", "bm3d", "fdncnn_color", "ectv_fdncnn"] if m in outputs]
    fig, axs = plt.subplots(3, len(methods) + 1, figsize=(13.5, 4.8))
    add_image(axs[0, 0], clean, "clean")
    add_image(axs[1, 0], abs_error(clean, noisy), "noisy error", cmap="magma")
    add_image(axs[2, 0], grad_diff(clean, noisy), "noisy grad diff", cmap="viridis")
    for col, method in enumerate(methods, start=1):
        out = outputs[method][0]
        add_image(axs[0, col], out, METHOD_LABELS.get(method, method))
        add_image(axs[1, col], abs_error(clean, out), "|error|", cmap="magma")
        add_image(axs[2, col], grad_diff(clean, out), "grad diff", cmap="viridis")
    fig.suptitle(f"Synthetic color denoising, sigma={sigma}, seed={seed}", fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.01, top=0.92, wspace=0.015, hspace=0.09)
    fig.savefig(FIGS / "fig4_synthetic_color_visual.pdf", bbox_inches="tight")
    plt.close(fig)


def make_real_panel(clean, noisy, outputs, name) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(3.45, 3.25))
    add_image(axs[0, 0], clean, "GT")
    add_image(axs[0, 1], noisy, "Noisy")
    add_image(axs[0, 2], outputs.get("fdncnn_color", (noisy,))[0], "FDnCNN")
    add_image(axs[1, 0], outputs.get("ectv_fidelity", (noisy,))[0], "ECTV")
    add_image(axs[1, 1], outputs.get("bm3d", (noisy,))[0], "BM3D")
    add_image(axs[1, 2], outputs.get("ectv_fdncnn", (noisy,))[0], "Cert-FD")
    add_image(axs[2, 0], abs_error(clean, noisy), "Noisy err", cmap="magma")
    add_image(axs[2, 1], abs_error(clean, outputs.get("fdncnn_color", (noisy,))[0]), "FDN err", cmap="magma")
    add_image(axs[2, 2], abs_error(clean, outputs.get("ectv_fdncnn", (noisy,))[0]), "Cert err", cmap="magma")
    fig.suptitle("PolyU real-noise audit", fontsize=7.5, y=0.995)
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.004, top=0.91, wspace=-0.35, hspace=0.08)
    fig.savefig(FIGS / "fig5_real_camera_visual.pdf", bbox_inches="tight")
    plt.close(fig)


def make_energy_panel(ectv_result, tv_result, name: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].plot(ectv_result.tv_curve, label="ECTV")
    axs[0].plot(tv_result.tv_curve, label="Full TV", alpha=0.8)
    axs[0].set_title("TV energy")
    axs[0].set_xlabel("iteration")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    removed = np.maximum(0, ectv_result.tv_curve[0] - np.array(ectv_result.tv_curve))
    axs[1].plot(removed, label="removed variation")
    axs[1].axhline(ectv_result.budget, color="crimson", ls="--", label="budget")
    axs[1].set_title("ECTV stopping budget")
    axs[1].set_xlabel("iteration")
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=8)
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_energy_iteration_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def make_video_panel(clean_frames, noisy_frames, method_outputs, name) -> None:
    methods = [m for m in ["noisy", "tv_maxiter", "tv_chambolle_val", "ectv", "ectv_fidelity", "wavelet_bayes", "bm3d", "fdncnn_color", "ectv_fdncnn"] if m in method_outputs]
    row_labels = ["Clean"] + [METHOD_LABELS.get(method, method) for method in methods]
    row_labels = [label.replace("FDnCNN-Color", "FDnCNN").replace("ECTV-Cert-FDnCNN", "Cert-FDnCNN") for label in row_labels]
    fig, axs = plt.subplots(
        len(methods) + 1,
        4,
        figsize=(3.45, 5.9),
        gridspec_kw={"width_ratios": [0.62, 1.0, 1.0, 1.0]},
    )
    idxs = [1, 3, 5]
    for row, label in enumerate(row_labels):
        label_ax = axs[row, 0]
        label_ax.axis("off")
        label_ax.text(0.98, 0.5, label, ha="right", va="center", fontsize=5.6)
    for col, idx in enumerate(idxs):
        ax = axs[0, col + 1]
        ax.imshow(np.clip(clean_frames[idx], 0, 1))
        ax.set_title(f"t={idx}", fontsize=6.4, pad=1.0)
        ax.axis("off")
    for row, method in enumerate(methods, start=1):
        outs = method_outputs[method]
        for col, idx in enumerate(idxs):
            ax = axs[row, col + 1]
            ax.imshow(np.clip(outs[idx], 0, 1))
            ax.axis("off")
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.985, wspace=0.012, hspace=0.035)
    fig.savefig(FIGS / "fig6_video_temporal.pdf", bbox_inches="tight")
    plt.close(fig)


def make_matching_panel(df: pd.DataFrame) -> None:
    plot_df = df[df["detector"] == "orb"].copy()
    summary = plot_df.groupby("method")[["inliers", "corner_error"]].mean().reindex(DOWNSTREAM_METHOD_ORDER)
    summary = summary.dropna(how="all")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    summary["inliers"].plot(kind="bar", ax=axs[0], rot=35, color="#4c78a8")
    axs[0].set_ylabel("RANSAC inliers")
    axs[0].set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in axs[0].get_xticklabels()], ha="right")
    axs[0].grid(axis="y", alpha=0.25)
    summary["corner_error"].plot(kind="bar", ax=axs[1], rot=35, color="#f58518")
    axs[1].set_ylabel("corner error (px)")
    axs[1].set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in axs[1].get_xticklabels()], ha="right")
    axs[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_downstream_matching.pdf", bbox_inches="tight")
    plt.close(fig)


def make_tracking_panel(df: pd.DataFrame) -> None:
    if df.empty:
        return
    summary = df.groupby("method")[["klt_epe", "farneback_epe"]].mean().reindex(DOWNSTREAM_METHOD_ORDER).dropna(how="all")
    ax = summary.plot(kind="bar", figsize=(7, 3), rot=35)
    ax.set_ylabel("endpoint error (px)")
    ax.set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGS / "fig11_downstream_tracking.pdf", bbox_inches="tight")
    plt.close(fig)


def make_edge_panel(clean: np.ndarray, outputs, name: str) -> None:
    methods = [m for m in ["noisy", "tv_maxiter", "tv_chambolle_val", "ectv", "wavelet_bayes"] if m in outputs]
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    axs = axs.ravel()
    ref_edges = canny(to_gray(clean).astype(np.float32), sigma=1.0, low_threshold=0.10, high_threshold=0.20)
    add_image(axs[0], ref_edges.astype(np.float32), "clean Canny", cmap="gray")
    for ax, method in zip(axs[1:], methods):
        add_image(ax, edge_agreement_map(clean, outputs[method][0]), METHOD_LABELS.get(method, method))
    for ax in axs[len(methods) + 1 :]:
        ax.axis("off")
    fig.suptitle(f"Edge consistency: {name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_downstream_edge.pdf", bbox_inches="tight")
    plt.close(fig)


def make_compression_panel(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    for method, group in df.groupby("method"):
        agg = group.groupby("quality")[["bpp", "psnr"]].mean().sort_index()
        ax.plot(agg["bpp"], agg["psnr"], marker="o", label=METHOD_LABELS.get(method, method))
    ax.set_xlabel("bits per pixel")
    ax.set_ylabel("PSNR after JPEG")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_downstream_compression.pdf", bbox_inches="tight")
    plt.close(fig)


def make_ablation_panel(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    p_df = df[df["ablation"] == "p_sweep"].groupby("p")[["psnr", "ssim", "grad_corr", "iterations"]].mean()
    axs[0, 0].plot(p_df.index, p_df["psnr"], marker="o", label="PSNR")
    ax2 = axs[0, 0].twinx()
    ax2.plot(p_df.index, p_df["iterations"], marker="s", color="crimson", label="iterations")
    axs[0, 0].set_title("p sweep")
    axs[0, 0].set_xlabel("p")
    axs[0, 0].set_ylabel("PSNR")
    ax2.set_ylabel("iterations")
    axs[0, 0].grid(alpha=0.25)

    lam_df = df[df["ablation"] == "lambda_sweep"].groupby("lambda")[["psnr", "ssim", "grad_corr"]].mean()
    axs[0, 1].plot(lam_df.index, lam_df["psnr"], marker="o", label="PSNR")
    axs[0, 1].plot(lam_df.index, lam_df["ssim"], marker="s", label="SSIM")
    axs[0, 1].set_title("lambda sweep")
    axs[0, 1].set_xlabel("lambda")
    axs[0, 1].grid(alpha=0.25)
    axs[0, 1].legend(fontsize=8)

    map_df = df[df["ablation"] == "mapping_gamma"].groupby(["mapping", "gamma"])[["psnr", "budget"]].mean().reset_index()
    for mapping, group in map_df.groupby("mapping"):
        axs[1, 0].plot(group["gamma"], group["psnr"], marker="o", label=mapping)
    axs[1, 0].set_title("budget mapping")
    axs[1, 0].set_xlabel("gamma")
    axs[1, 0].set_ylabel("PSNR")
    axs[1, 0].grid(alpha=0.25)
    axs[1, 0].legend(fontsize=8)

    stop_df = df[df["ablation"] == "stopping"].groupby("method")[["psnr", "iterations"]].mean()
    stop_df = stop_df.reindex(["ectv_budget", "tv_fixedk_val", "tv_discrepancy", "tv_chambolle_val", "tv_maxiter", "tv_oracle"])
    stop_df["psnr"].plot(kind="bar", ax=axs[1, 1], color="#4c78a8", rot=20)
    axs[1, 1].set_title("stopping criterion")
    axs[1, 1].set_ylabel("PSNR")
    axs[1, 1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGS / "fig9_ablation_psweep.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(dfs: Dict[str, pd.DataFrame]) -> None:
    def fmt(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{float(value):.3f}"

    def fmt_mean_ci(group: pd.DataFrame, col: str) -> str:
        vals = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return "--"
        arr = vals.to_numpy(dtype=float)
        mean = float(arr.mean())
        if len(arr) <= 1:
            return f"{mean:.3f} $\\pm$ 0.000"
        rng = np.random.default_rng(20260521)
        samples = rng.choice(arr, size=(1000, len(arr)), replace=True).mean(axis=1)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        half_width = float(max(mean - lo, hi - mean))
        return f"{mean:.3f} $\\pm$ {half_width:.3f}"

    def write_rows(headers: List[str], rows: List[List[str]], path: Path, align: str | None = None) -> None:
        if align is None:
            align = "l" + "c" * (len(headers) - 1)
        lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")
        for row in rows:
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_restoration_table(df: pd.DataFrame, path: Path) -> None:
        rows = []
        for method in METHOD_ORDER:
            group = df[df["method"] == method]
            if group.empty:
                continue
            rows.append(
                [
                    method_label(method),
                    fmt_mean_ci(group, "psnr"),
                    fmt_mean_ci(group, "ssim"),
                    fmt_mean_ci(group, "grad_corr"),
                    fmt_mean_ci(group, "iterations"),
                    fmt_mean_ci(group, "runtime"),
                ]
            )
        write_rows(["Method", "PSNR", "SSIM", "Grad. corr.", "Iter.", "Runtime (s)"], rows, path)

    dfs = {name: df.replace([np.inf, -np.inf], np.nan) for name, df in dfs.items()}
    write_restoration_table(dfs["synthetic"], GENERATED / "tab_image_aggregate.tex")
    write_restoration_table(dfs["real"], GENERATED / "tab_real_camera_aggregate.tex")

    sigma_rows = []
    for method in METHOD_ORDER:
        for sigma in sorted(dfs["synthetic"]["sigma255"].dropna().unique()):
            group = dfs["synthetic"][(dfs["synthetic"]["method"] == method) & (dfs["synthetic"]["sigma255"] == sigma)]
            if group.empty:
                continue
            sigma_rows.append([method_label(method), f"{int(sigma)}", fmt_mean_ci(group, "psnr"), fmt_mean_ci(group, "ssim"), fmt_mean_ci(group, "grad_corr")])
    write_rows(["Method", "$\\sigma/255$", "PSNR", "SSIM", "Grad. corr."], sigma_rows, GENERATED / "tab_image_by_sigma.tex")

    video_rows = []
    video = dfs["video"]
    for method in METHOD_ORDER:
        group = video[video["method"] == method]
        if group.empty:
            continue
        video_rows.append(
            [
                method_label(method),
                fmt(group["psnr"].mean()),
                fmt(group["ssim"].mean()),
                fmt(group["temporal_residual_fluctuation"].mean()),
            ]
        )
    write_rows(["Method", "PSNR", "SSIM", "TRF"], video_rows, GENERATED / "tab_video_summary.tex")

    feature = dfs["matching"].groupby(["detector", "method"])[["matches", "inliers", "inlier_ratio", "corner_error", "homography_success"]].mean()
    feature_rows = []
    for detector in ["orb", "sift"]:
        if detector not in feature.index.get_level_values(0):
            continue
        for method in DOWNSTREAM_METHOD_ORDER:
            key = (detector, method)
            if key not in feature.index:
                continue
            row = feature.loc[key]
            feature_rows.append(
                [
                    detector.upper(),
                    method_label(method),
                    fmt(row.get("matches")),
                    fmt(row.get("inliers")),
                    fmt(row.get("inlier_ratio")),
                    fmt(row.get("corner_error")),
                    fmt(row.get("homography_success")),
                ]
            )
    write_rows(
        ["Detector", "Method", "Matches", "Inliers", "Inlier ratio", "Corner err.", "Success"],
        feature_rows,
        GENERATED / "tab_downstream_feature_summary.tex",
    )

    edge = dfs["edge"].groupby("method")[["edge_precision", "edge_recall", "edge_f1", "edge_f1_tol2", "edge_density"]].mean()
    edge_rows = []
    for method in DOWNSTREAM_METHOD_ORDER:
        if method not in edge.index:
            continue
        row = edge.loc[method]
        edge_rows.append(
            [
                method_label(method),
                fmt(row.get("edge_precision")),
                fmt(row.get("edge_recall")),
                fmt(row.get("edge_f1")),
                fmt(row.get("edge_f1_tol2")),
                fmt(row.get("edge_density")),
            ]
        )
    write_rows(
        ["Method", "Edge P", "Edge R", "Edge F1-1px", "Edge F1-2px", "Edge density"],
        edge_rows,
        GENERATED / "tab_downstream_edge_summary.tex",
    )

    orb = dfs["matching"][dfs["matching"]["detector"] == "orb"].groupby("method")[["inliers", "corner_error", "homography_success"]].mean()
    structure = orb.join(edge[["edge_f1", "edge_f1_tol2"]], how="outer")
    structure_rows = []
    for method in DOWNSTREAM_METHOD_ORDER:
        if method not in structure.index:
            continue
        row = structure.loc[method]
        structure_rows.append([method_label(method), fmt(row.get("inliers")), fmt(row.get("corner_error")), fmt(row.get("homography_success")), fmt(row.get("edge_f1")), fmt(row.get("edge_f1_tol2"))])
    write_rows(["Method", "ORB inliers", "ORB corner err.", "ORB success", "Edge F1-1px", "Edge F1-2px"], structure_rows, GENERATED / "tab_downstream_structure_summary.tex")

    tracking = dfs["tracking"].groupby("method")[["klt_epe", "klt_survival", "klt_bad1", "farneback_epe", "farneback_bad1"]].mean()
    tracking_rows = []
    for method in DOWNSTREAM_METHOD_ORDER:
        if method not in tracking.index:
            continue
        row = tracking.loc[method]
        tracking_rows.append([method_label(method), fmt(row.get("klt_epe")), fmt(row.get("klt_survival")), fmt(row.get("klt_bad1")), fmt(row.get("farneback_epe")), fmt(row.get("farneback_bad1"))])
    write_rows(["Method", "KLT EPE", "KLT survival", "KLT bad-1", "Flow EPE", "Flow bad-1"], tracking_rows, GENERATED / "tab_downstream_tracking_summary.tex")

    comp = dfs["compression"].groupby("method")[["bpp", "psnr", "ssim"]].mean()

    def series_value(series: pd.Series, method: str) -> str:
        if method not in series.index:
            return "--"
        return fmt(series.loc[method])

    def best_entry(series: pd.Series, higher_is_better: bool = True) -> str:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            return "--"
        best_method = vals.idxmax() if higher_is_better else vals.idxmin()
        return f"{method_label(str(best_method))} ({fmt(vals.loc[best_method])})"

    profile_rows = []
    def add_profile_row(pipeline: str, metric: str, series: pd.Series, higher_is_better: bool = True) -> None:
        profile_rows.append(
            [
                pipeline,
                metric,
                series_value(series, "noisy"),
                series_value(series, "ectv"),
                series_value(series, "ectv_fidelity"),
                series_value(series, "ectv_fdncnn"),
                best_entry(series, higher_is_better=higher_is_better),
            ]
        )

    if "orb" in feature.index.get_level_values(0):
        orb_feature = feature.xs("orb")
        add_profile_row("ORB registration", "Inliers $\\uparrow$", orb_feature["inliers"], True)
        add_profile_row("ORB registration", "Corner err. $\\downarrow$", orb_feature["corner_error"], False)
    if "sift" in feature.index.get_level_values(0):
        sift_feature = feature.xs("sift")
        add_profile_row("SIFT registration", "Inliers $\\uparrow$", sift_feature["inliers"], True)
        add_profile_row("SIFT registration", "Corner err. $\\downarrow$", sift_feature["corner_error"], False)
    add_profile_row("Canny edges", "F1-1px $\\uparrow$", edge["edge_f1"], True)
    add_profile_row("Canny edges", "Recall $\\uparrow$", edge["edge_recall"], True)
    add_profile_row("KLT tracking", "EPE $\\downarrow$", tracking["klt_epe"], False)
    add_profile_row("Farneback flow", "EPE $\\downarrow$", tracking["farneback_epe"], False)
    add_profile_row("JPEG compression", "JPEG PSNR $\\uparrow$", comp["psnr"], True)
    write_rows(
        ["Pipeline", "Metric", "Noisy", "ECTV-S", "ECTV-F", "ECTV-L", "Best"],
        profile_rows,
        GENERATED / "tab_task_advantage_summary.tex",
        align=r"lp{0.16\textwidth}ccccc",
    )

    comp_rows = []
    for method in DOWNSTREAM_METHOD_ORDER:
        if method not in comp.index:
            continue
        row = comp.loc[method]
        comp_rows.append([method_label(method), fmt(row["bpp"]), fmt(row["psnr"]), fmt(row["ssim"])])
    write_rows(["Method", "bpp", "JPEG PSNR", "JPEG SSIM"], comp_rows, GENERATED / "tab_downstream_compression_summary.tex")

    # Keep the previous combined downstream table for compatibility with older drafts.
    downstream = structure.join(tracking[["klt_epe", "farneback_epe"]], how="outer").join(comp, rsuffix="_jpeg")
    downstream_rows = []
    for method in DOWNSTREAM_METHOD_ORDER:
        if method not in downstream.index:
            continue
        row = downstream.loc[method]
        downstream_rows.append(
            [
                method_label(method),
                fmt(row.get("inliers")),
                fmt(row.get("corner_error")),
                fmt(row.get("edge_f1")),
                fmt(row.get("klt_epe")),
                fmt(row.get("bpp")),
                fmt(row.get("psnr")),
                fmt(row.get("ssim")),
            ]
        )
    write_rows(["Method", "ORB inliers", "ORB err.", "Edge F1", "KLT EPE", "bpp", "JPEG PSNR", "JPEG SSIM"], downstream_rows, GENERATED / "tab_downstream_summary.tex")

    ablation = dfs["ablation"].groupby(["ablation", "method"])[["psnr", "ssim", "grad_corr", "iterations", "budget"]].mean()
    ablation_rows = []
    for (ablation_name, method), row in ablation.iterrows():
        ablation_rows.append(
            [
                str(ablation_name).replace("_", r"\_"),
                method_label(method),
                fmt(row["psnr"]),
                fmt(row["ssim"]),
                fmt(row["grad_corr"]),
                fmt(row["iterations"]),
                fmt(row["budget"]),
            ]
        )
    write_rows(["Ablation", "Method", "PSNR", "SSIM", "Grad. corr.", "Iter.", "Budget"], ablation_rows, GENERATED / "tab_ablation_summary.tex")

    baseline_rows = []
    for method in METHOD_ORDER:
        source, rule = BASELINE_DESCRIPTIONS.get(method, ("", ""))
        baseline_rows.append([method_label(method), source.replace("_", r"\_"), rule.replace("_", r"\_")])
    write_rows(["Method", "Implementation", "Parameter rule / role"], baseline_rows, GENERATED / "tab_baseline_suite.tex", align=r"lp{0.22\textwidth}p{0.55\textwidth}")

    workload_rows = []
    synth_cases = dfs["synthetic"].drop_duplicates(["dataset", "image", "sigma255", "seed"])
    workload_rows.append(["Synthetic Kodak color", str(synth_cases[["dataset", "image"]].drop_duplicates().shape[0]), "3 noise levels $\\times$ 3 seeds", str(len(dfs["synthetic"]))])
    real_cases = dfs["real"].drop_duplicates(["dataset", "image"])
    workload_rows.append(["PolyU real camera", str(len(real_cases)), "blind sigma estimate", str(len(dfs["real"]))])
    video_cases = dfs["video"].drop_duplicates(["image"])
    workload_rows.append(["Frame sequence", str(len(video_cases)), "8 translated frames", str(len(dfs["video"]))])
    feat_cases = dfs["matching"].drop_duplicates(["dataset", "image", "sigma255", "detector"])
    feat_sigmas = sorted(int(v) for v in dfs["matching"]["sigma255"].dropna().unique()) if "sigma255" in dfs["matching"] else []
    workload_rows.append(["Registration", str(len(feat_cases)), f"ORB/SIFT known translation, $\\sigma={feat_sigmas}$", str(len(dfs["matching"]))])
    track_cases = dfs["tracking"].drop_duplicates(["dataset", "image", "sigma255", "transition"])
    track_sigmas = sorted(int(v) for v in dfs["tracking"]["sigma255"].dropna().unique()) if "sigma255" in dfs["tracking"] else []
    workload_rows.append(["Tracking/flow", str(len(track_cases)), f"KLT/Farneback EPE, $\\sigma={track_sigmas}$", str(len(dfs["tracking"]))])
    comp_cases = dfs["compression"].drop_duplicates(["dataset", "image", "quality"])
    workload_rows.append(["JPEG RD", str(len(comp_cases)), "5 qualities" if dfs["compression"]["quality"].nunique() >= 5 else "2 qualities", str(len(dfs["compression"]))])
    write_rows(["Experiment", "Cases", "Protocol", "Method evals"], workload_rows, GENERATED / "tab_workload_summary.tex", align="lccc")


def write_manifest(images, calib_images, pairs, args, baseline_config: Dict[str, float | int | str], dfs: Dict[str, pd.DataFrame]) -> None:
    method_names = [m for m in METHOD_ORDER if any(m in set(df.get("method", pd.Series(dtype=str))) for df in dfs.values())]
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {
            "kodak24_downloaded_or_available": len([i for i in images if i[0] == "kodak24"]),
            "legacy_calibration_images": len(calib_images),
            "polyu_pairs": len(pairs),
            "polyu_source": "polyu_full" if pairs and "polyu_full" in pairs[0][1].parts else "polyu_local",
        },
        "methods": method_names,
        "parameters": vars(args),
        "baseline_config": baseline_config,
        "reproducibility": {
            "command": f"python experiments/run_tmm_pipeline.py --max-iter {args.max_iter}" + (" --quick" if args.quick else ""),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "opencv": cv2.__version__,
            "sift_available": hasattr(cv2, "SIFT_create"),
            "bm3d_available": bm3d is not None,
            "h5py_available": h5py is not None,
            "fdncnn_color_available": h5py is not None and FDNCNN_COLOR_MODEL.exists(),
            "fdncnn_color_model": str(FDNCNN_COLOR_MODEL.relative_to(ROOT)) if FDNCNN_COLOR_MODEL.exists() else "",
            "synthetic_sigmas_255": [15, 25] if args.quick else [15, 25, 50],
            "synthetic_seeds": [0] if args.quick else [0, 1, 2],
            "resize_max_side": 256,
            "ectv_lambda": 0.08,
            "ectv_max_iter": args.max_iter,
            "p_tau": 25.0 / 255.0,
            "p_rho": 2.0,
            "p_min": 0.501,
            "budget_alpha": 1.0 / 3.0,
            "budget_gamma_structure": float(baseline_config["ectv_gamma_low"]),
            "budget_gamma_fidelity": float(baseline_config["ectv_gamma_high"]),
            "ectv_fidelity_low_lam_alpha": [float(baseline_config["ectv_fidelity_low_lam"]), float(baseline_config["ectv_fidelity_low_alpha"])],
            "ectv_fidelity_mid_lam_alpha": [float(baseline_config["ectv_fidelity_mid_lam"]), float(baseline_config["ectv_fidelity_mid_alpha"])],
            "ectv_fidelity_high_lam_alpha": [float(baseline_config["ectv_fidelity_high_lam"]), float(baseline_config["ectv_fidelity_high_alpha"])],
            "nlm_patch_size": 5,
            "nlm_patch_distance": 6,
            "nlm_h_scale": 0.85,
            "pm_iterations": int(baseline_config["pm_iterations"]),
            "pm_kappa_scale": float(baseline_config["pm_kappa_scale"]),
            "edge_canny_sigma": 1.0,
            "edge_canny_low": 0.10,
            "edge_canny_high": 0.20,
            "edge_tolerance_px": [1, 2],
            "downstream_feature_edge_sigmas_255": [25, 50],
            "downstream_tracking_sigmas_255": [20, 50],
            "jpeg_qualities": [30, 50] if args.quick else [10, 20, 30, 50, 70],
            "tracking_sequence_frames": 8,
            "tracking_step_dx_dy": [2.0, 1.0],
        },
        "notes": [
            "Kodak images are downloaded from r0k.us when reachable.",
            "Legacy local images are used only for baseline calibration when available; Kodak24 is the synthetic test set.",
            "Processed Kodak and PolyU caches are used when present; otherwise Kodak is downloaded and PolyU is prepared from local raw files or fallback project crops.",
            "BM3D-RGB is reported only when the optional bm3d Python package imports successfully.",
            "FDnCNN-Color is reported in image, sequence, and downstream tables when h5py and the official FDnCNN_color.mat model are available.",
            "ECTV-Cert-FDnCNN certifies the FDnCNN-Color proposal with the ECTV-Fidelity removed-TV budget and projects it to the budget boundary when needed.",
            "Synthetic sequence is generated from image translations for controlled multimedia pipeline testing.",
            "Edge extraction is a fixed Canny consistency proxy against clean-image Canny pseudo-references.",
            "Feature registration evaluates ORB and SIFT when available using known translated pairs at two noise levels and corner reprojection error.",
            "Tracking/flow evaluates KLT and Farneback endpoint error on known translated sequences at two noise levels.",
            "Ablation rows are generated from a fixed subset and written to results/ablation.csv.",
        ],
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a smaller deterministic suite.")
    parser.add_argument("--max-iter", type=int, default=80)
    args = parser.parse_args()

    ensure_dirs()
    make_budget_curves()

    kodak_paths = download_kodak()
    images = [("kodak24", p.stem, p) for p in kodak_paths]
    calib_images = [("legacy", p.stem, p) for p in legacy_images()]
    if not calib_images:
        calib_images = images[:4]
    if args.quick:
        images = images[:4]
        calib_images = calib_images[:2]
    if not images:
        raise RuntimeError("No usable images found for synthetic experiments.")
    baseline_config = calibrate_baselines(calib_images, quick=args.quick)
    print("Baseline calibration:", baseline_config)

    pairs = prepare_local_polyu()
    if not pairs:
        raise RuntimeError("No local PolyU noisy/mean pairs found.")
    if args.quick:
        pairs = pairs[:8]

    dfs = {
        "synthetic": run_synthetic(images, max_iter=args.max_iter, quick=args.quick, baseline_config=baseline_config),
        "real": run_real_noise(pairs, max_iter=args.max_iter, baseline_config=baseline_config),
        "video": run_video(images, max_iter=args.max_iter, baseline_config=baseline_config),
        "matching": run_feature_matching(images, max_iter=args.max_iter, quick=args.quick, baseline_config=baseline_config),
        "edge": run_edge_extraction(images, max_iter=args.max_iter, quick=args.quick, baseline_config=baseline_config),
        "tracking": run_tracking(images, max_iter=args.max_iter, quick=args.quick, baseline_config=baseline_config),
        "compression": run_compression(images, max_iter=args.max_iter, quick=args.quick, baseline_config=baseline_config),
        "ablation": run_ablation(images, max_iter=args.max_iter, quick=args.quick, baseline_config=baseline_config),
    }
    write_latex_tables(dfs)
    write_manifest(images, calib_images, pairs, args, baseline_config, dfs)
    print("Wrote results to", RESULTS)
    print("Wrote figures to", FIGS)
    print("Wrote LaTeX tables to", GENERATED)


if __name__ == "__main__":
    main()
