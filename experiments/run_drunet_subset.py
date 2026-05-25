from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EARLY_DEP_PATHS = [
    os.environ.get("ECTV_TORCH_DEPS", "").strip(),
    str(ROOT / ".ectv_pydeps_torch"),
]
for raw_path in EARLY_DEP_PATHS:
    if raw_path:
        dep_path = Path(raw_path).expanduser()
        if dep_path.exists() and str(dep_path) not in sys.path:
            sys.path.insert(0, str(dep_path))

import numpy as np
import pandas as pd
from PIL import Image

RESULTS = ROOT / "results"
GENERATED = ROOT / "generated"
DATA = ROOT / "data"
DEFAULT_TORCH_DEPS = [
    ROOT / ".ectv_pydeps_torch",
    Path("/home/fuyue/.cache/ectv_pydeps_torch"),
]
KAIR_MINIMAL = DATA / "models" / "kair_minimal"
DRUNET_WEIGHTS = DATA / "models" / "drunet_color.pth"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ectv import certify_candidate, tv_energy


def torch_dependency_paths(extra_path: str | None = None) -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("ECTV_TORCH_DEPS", "").strip()
    for raw in [extra_path, env_path]:
        if raw:
            paths.append(Path(raw).expanduser())
    paths.extend(DEFAULT_TORCH_DEPS)
    return paths


def add_torch_paths(extra_path: str | None = None) -> None:
    for path in torch_dependency_paths(extra_path):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if str(KAIR_MINIMAL) not in sys.path:
        sys.path.insert(0, str(KAIR_MINIMAL))


def read_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def add_gaussian(clean: np.ndarray, sigma255: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, sigma255 / 255.0, clean.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def gradient_correlation(clean: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    def grad_mag(x: np.ndarray) -> np.ndarray:
        gray = rgb_to_gray(x)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        gy[:-1, :] = gray[1:, :] - gray[:-1, :]
        return np.sqrt(gx * gx + gy * gy)

    gc = grad_mag(clean)
    go = grad_mag(out)
    denom = float(np.linalg.norm(gc.ravel()) * np.linalg.norm(go.ravel()) + 1e-12)
    corr = float(np.dot(gc.ravel(), go.ravel()) / denom)
    mae = float(np.mean(np.abs(gc - go)))
    return corr, mae


def ssim_channel(x: np.ndarray, y: np.ndarray, win: int = 7) -> float:
    from scipy.ndimage import uniform_filter

    c1 = 0.01**2
    c2 = 0.03**2
    ux = uniform_filter(x, size=win, mode="reflect")
    uy = uniform_filter(y, size=win, mode="reflect")
    uxx = uniform_filter(x * x, size=win, mode="reflect")
    uyy = uniform_filter(y * y, size=win, mode="reflect")
    uxy = uniform_filter(x * y, size=win, mode="reflect")
    vx = np.maximum(0.0, uxx - ux * ux)
    vy = np.maximum(0.0, uyy - uy * uy)
    cov = uxy - ux * uy
    num = (2.0 * ux * uy + c1) * (2.0 * cov + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def metric_row(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray) -> dict[str, float]:
    try:
        from skimage import color
        from skimage.filters import sobel
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        ch_axis = -1 if clean.ndim == 3 else None
        min_side = min(clean.shape[:2])
        win_size = 7 if min_side >= 7 else max(3, min_side // 2 * 2 - 1)
        psnr = float(peak_signal_noise_ratio(clean, out, data_range=1.0))
        ssim = float(structural_similarity(clean, out, channel_axis=ch_axis, data_range=1.0, win_size=win_size))
        clean_gray = color.rgb2gray(clean) if clean.ndim == 3 else clean
        out_gray = color.rgb2gray(out) if out.ndim == 3 else out
        gc = sobel(clean_gray)
        go = sobel(out_gray)
        denom = float(np.linalg.norm(gc.ravel()) * np.linalg.norm(go.ravel()) + 1e-12)
        grad_corr = float(np.dot(gc.ravel(), go.ravel()) / denom)
        grad_mae = float(np.mean(np.abs(gc - go)))
    except Exception:
        mse = float(np.mean((clean - out) ** 2))
        psnr = 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))
        ssim = float(np.mean([ssim_channel(clean[..., c], out[..., c]) for c in range(3)]))
        grad_corr, grad_mae = gradient_correlation(clean, out)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "grad_corr": grad_corr,
        "grad_mae": grad_mae,
        "tv_input": tv_energy(noisy),
        "tv_output": tv_energy(out),
    }


def fidelity_alpha_gamma(sigma_est: float) -> tuple[float, float]:
    sigma255 = 255.0 * max(float(sigma_est), 0.0)
    if sigma255 < 20.0:
        return 1.0 / 3.0, 0.32
    if sigma255 < 40.0:
        return 1.0 / 3.0, 0.32
    return 0.75, 0.32


def load_drunet(torch_deps: str | None = None):
    add_torch_paths(torch_deps)
    import torch
    import torch.nn.functional as F
    from network_unet import UNetRes

    if not DRUNET_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing DRUNet weights: {DRUNET_WEIGHTS}")

    model = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        bias=False,
    )
    state = torch.load(DRUNET_WEIGHTS, map_location="cpu")
    if isinstance(state, dict):
        for key in ["params_ema", "params", "state_dict", "model_state_dict"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict):
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    return model, torch, F


def denoise_drunet(model, torch, F, noisy: np.ndarray, sigma_est: float) -> np.ndarray:
    h, w = noisy.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    sigma_map = np.full((h, w, 1), float(max(sigma_est, 0.0)), dtype=np.float32)
    x = np.concatenate([noisy.astype(np.float32), sigma_map], axis=2)
    tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        out = model(tensor).clamp_(0.0, 1.0)
    out = out[..., :h, :w].squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def select_synthetic_cases(count: int) -> list[dict[str, object]]:
    rows = []
    for path in sorted((DATA / "processed" / "kodak24").glob("kodim*.png"))[:count]:
        clean = read_image(path)
        sigma255 = 25.0
        seed = 0
        rows.append(
            {
                "task": "synthetic_image",
                "dataset": "kodak24",
                "image": path.stem,
                "sigma255": sigma255,
                "seed": seed,
                "clean": clean,
                "noisy": add_gaussian(clean, sigma255, seed),
            }
        )
    return rows


def select_polyu_cases(count: int) -> list[dict[str, object]]:
    existing = pd.read_csv(RESULTS / "real_camera.csv")
    noisy_rows = existing[existing["method"] == "noisy"].drop_duplicates(["image"]).head(count)
    rows = []
    for _, source in noisy_rows.iterrows():
        name = str(source["image"])
        noisy_path = DATA / "processed" / "polyu_full" / f"{name}_noisy.png"
        gt_path = DATA / "processed" / "polyu_full" / f"{name}_gt.png"
        if not noisy_path.exists() or not gt_path.exists():
            continue
        rows.append(
            {
                "task": "real_camera",
                "dataset": "polyu_full",
                "image": name,
                "sigma255": float(source["sigma255"]),
                "seed": -1,
                "clean": read_image(gt_path),
                "noisy": read_image(noisy_path),
            }
        )
    return rows


def append_existing_references(rows: list[dict[str, object]], source_csv: Path, method_names: list[str]) -> list[dict[str, object]]:
    source = pd.read_csv(source_csv)
    key_cols = ["task", "dataset", "image", "sigma255", "seed"]
    wanted = pd.DataFrame([{col: row[col] for col in key_cols} for row in rows])
    if wanted.empty:
        return []
    merged = source.merge(wanted, on=key_cols, how="inner")
    keep = merged[merged["method"].isin(method_names)].copy()
    return keep.to_dict("records")


def run_cases(cases: list[dict[str, object]], model, torch, F) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases, 1):
        clean = np.asarray(case["clean"], dtype=np.float32)
        noisy = np.asarray(case["noisy"], dtype=np.float32)
        sigma_est = float(case["sigma255"]) / 255.0
        print(f"[drunet] {idx}/{len(cases)} {case['dataset']} {case['image']} sigma255={case['sigma255']:.3f}", flush=True)
        t0 = time.perf_counter()
        drunet = denoise_drunet(model, torch, F, noisy, sigma_est)
        runtime = time.perf_counter() - t0
        base = {
            "task": case["task"],
            "dataset": case["dataset"],
            "image": case["image"],
            "sigma255": case["sigma255"],
            "seed": case["seed"],
        }
        metrics = metric_row(clean, noisy, drunet)
        removed = max(0.0, metrics["tv_input"] - metrics["tv_output"])
        out_rows.append({**base, "method": "drunet_color", **metrics, "iterations": 0, "runtime": runtime, "budget": np.nan, "removed": removed, "p": 0.0, "gamma": np.nan, "lambda": np.nan, "alpha": np.nan, "eta": np.nan, "accepted": np.nan})

        alpha, gamma = fidelity_alpha_gamma(sigma_est)
        cert_t0 = time.perf_counter()
        cert = certify_candidate(noisy, drunet, sigma_est=max(sigma_est, 1e-6), alpha=alpha, gamma=gamma)
        cert_metrics = metric_row(clean, noisy, cert.image)
        out_rows.append(
            {
                **base,
                "method": "ectv_drunet",
                **cert_metrics,
                "iterations": 0,
                "runtime": runtime + (time.perf_counter() - cert_t0),
                "budget": cert.budget,
                "removed": cert.removed,
                "p": cert.p,
                "gamma": gamma,
                "lambda": np.nan,
                "alpha": alpha,
                "eta": cert.eta,
                "accepted": 1.0 if cert.accepted_without_projection else 0.0,
            }
        )
    return out_rows


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def write_latex_table(df: pd.DataFrame) -> None:
    labels = {
        "noisy": "Noisy",
        "fdncnn_color": "FDnCNN-Color",
        "ectv_fdncnn": "ECTV-Cert-FDnCNN",
        "drunet_color": "DRUNet-Color",
        "ectv_drunet": "ECTV-Cert-DRUNet",
    }
    dataset_labels = {
        "synthetic_image": "Kodak24 slice",
        "real_camera": "PolyU100 audit",
    }
    rows: list[list[str]] = []
    method_order = ["noisy", "fdncnn_color", "drunet_color", "ectv_fdncnn", "ectv_drunet"]
    for task in ["synthetic_image", "real_camera"]:
        sub_task = df[df["task"] == task]
        if sub_task.empty:
            continue
        for method in method_order:
            group = sub_task[sub_task["method"] == method]
            if group.empty:
                continue
            ratio = group["removed"] / group["budget"].replace(0, np.nan)
            eta = group["eta"].astype(float) if "eta" in group else pd.Series([], dtype=float)
            full_accept = eta >= 1.0 - 1e-7
            no_op = eta <= 1e-7
            projection = ~(full_accept | no_op)
            rows.append(
                [
                    dataset_labels[task],
                    labels.get(method, method),
                    str(len(group)),
                    fmt(float(group["psnr"].mean())),
                    fmt(float(group["ssim"].mean()), 4),
                    fmt(float(group["grad_corr"].mean()), 4),
                    fmt(float(ratio.mean())) if method.startswith("ectv_") else "--",
                    fmt(100.0 * float(full_accept.mean()), 1) if method.startswith("ectv_") else "--",
                    fmt(100.0 * float(projection.mean()), 1) if method.startswith("ectv_") else "--",
                    fmt(100.0 * float(no_op.mean()), 1) if method.startswith("ectv_") else "--",
                    fmt(float(eta.median()), 4) if method.startswith("ectv_") else "--",
                    fmt(float(group["runtime"].mean()), 2),
                ]
            )
    lines = [
        r"\begin{tabular}{llcccccccccc}",
        r"\toprule",
        r"Dataset & Method & Cases & PSNR & SSIM & Grad. corr. & Removed/Budget & Kept (\%) & Proj. (\%) & No-op (\%) & Med. $\eta$ & Runtime (s) \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "tab_modern_baseline_subset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an executable DRUNet subset audit for the ECTV paper.")
    parser.add_argument("--kodak-count", type=int, default=4)
    parser.add_argument("--poly-count", type=int, default=8)
    parser.add_argument("--torch-deps", help="Optional directory containing torch dependencies if torch is not installed globally. ECTV_TORCH_DEPS is also honored.")
    parser.add_argument("--table-only", action="store_true", help="Regenerate the LaTeX table from results/modern_baseline_subset.csv without rerunning DRUNet.")
    args = parser.parse_args()

    if args.table_only:
        csv_path = RESULTS / "modern_baseline_subset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        write_latex_table(pd.read_csv(csv_path))
        print(f"Wrote {GENERATED / 'tab_modern_baseline_subset.tex'}")
        return 0

    synthetic_cases = select_synthetic_cases(args.kodak_count)
    polyu_cases = select_polyu_cases(args.poly_count)
    all_cases = synthetic_cases + polyu_cases
    if not all_cases:
        raise RuntimeError("No cases selected for DRUNet subset audit.")

    model, torch, F = load_drunet(args.torch_deps)
    rows = []
    rows.extend(append_existing_references(synthetic_cases, RESULTS / "synthetic_image.csv", ["noisy", "fdncnn_color", "ectv_fdncnn"]))
    rows.extend(append_existing_references(polyu_cases, RESULTS / "real_camera.csv", ["noisy", "fdncnn_color", "ectv_fdncnn"]))
    rows.extend(run_cases(all_cases, model, torch, F))

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "modern_baseline_subset.csv", index=False)
    write_latex_table(df)
    manifest = {
        "model": "DRUNet color",
        "weights": str(DRUNET_WEIGHTS.relative_to(ROOT)),
        "kair_minimal_files": [str((KAIR_MINIMAL / name).relative_to(ROOT)) for name in ["network_unet.py", "basicblock.py"]],
        "kodak_count": len(synthetic_cases),
        "poly_count": len(polyu_cases),
        "command": f"python experiments/run_drunet_subset.py --kodak-count {args.kodak_count} --poly-count {args.poly_count}",
        "torch_dependency_paths": [str(path) for path in torch_dependency_paths(args.torch_deps)],
        "note": "CPU fixed-slice DRUNet audit: Kodak24 at sigma=25 and seed=0 plus the selected PolyU real-noise pairs.",
    }
    (RESULTS / "modern_baseline_subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS / 'modern_baseline_subset.csv'}")
    print(f"Wrote {GENERATED / 'tab_modern_baseline_subset.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
