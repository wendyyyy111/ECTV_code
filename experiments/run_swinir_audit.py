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

DATA = ROOT / "data"
RESULTS = ROOT / "results"
GENERATED = ROOT / "generated"
SWINIR_CODE = DATA / "models" / "swinir_minimal"
SWINIR_WEIGHTS = DATA / "models" / "swinir_color_noise25.pth"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SWINIR_CODE) not in sys.path:
    sys.path.insert(0, str(SWINIR_CODE))

from experiments.run_drunet_subset import add_gaussian, append_existing_references, metric_row, read_image
from methods.ectv import certify_candidate, tv_energy


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def load_swinir():
    import torch
    from network_swinir import SwinIR

    if not SWINIR_WEIGHTS.exists():
        raise FileNotFoundError(SWINIR_WEIGHTS)

    model = SwinIR(
        upscale=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    state = torch.load(SWINIR_WEIGHTS, map_location="cpu")
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    return model, torch


def denoise_swinir(model, torch, noisy: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.transpose(noisy.astype(np.float32), (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor).clamp_(0.0, 1.0)
    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out_np.astype(np.float32), 0.0, 1.0)


def select_kodak_cases(count: int) -> list[dict[str, object]]:
    rows = []
    for path in sorted((DATA / "processed" / "kodak24").glob("kodim*.png"))[:count]:
        clean = read_image(path)
        rows.append(
            {
                "task": "synthetic_image",
                "dataset": "kodak24",
                "image": path.stem,
                "sigma255": 25.0,
                "seed": 0,
                "clean": clean,
                "noisy": add_gaussian(clean, 25.0, 0),
            }
        )
    return rows


def run_cases(cases: list[dict[str, object]], model, torch) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases, 1):
        clean = np.asarray(case["clean"], dtype=np.float32)
        noisy = np.asarray(case["noisy"], dtype=np.float32)
        print(f"[swinir] {idx}/{len(cases)} {case['image']} sigma255=25.000", flush=True)
        t0 = time.perf_counter()
        swinir = denoise_swinir(model, torch, noisy)
        runtime = time.perf_counter() - t0
        base = {
            "task": case["task"],
            "dataset": case["dataset"],
            "image": case["image"],
            "sigma255": case["sigma255"],
            "seed": case["seed"],
        }
        metrics = metric_row(clean, noisy, swinir)
        removed = max(0.0, metrics["tv_input"] - metrics["tv_output"])
        out_rows.append(
            {
                **base,
                "method": "swinir_color",
                **metrics,
                "iterations": 0,
                "runtime": runtime,
                "budget": np.nan,
                "removed": removed,
                "p": 0.0,
                "gamma": np.nan,
                "lambda": np.nan,
                "alpha": np.nan,
                "eta": np.nan,
                "accepted": np.nan,
            }
        )

        cert_t0 = time.perf_counter()
        cert = certify_candidate(noisy, swinir, sigma_est=25.0 / 255.0, alpha=1.0 / 3.0, gamma=0.32)
        cert_metrics = metric_row(clean, noisy, cert.image)
        out_rows.append(
            {
                **base,
                "method": "ectv_swinir",
                **cert_metrics,
                "iterations": 0,
                "runtime": runtime + (time.perf_counter() - cert_t0),
                "budget": cert.budget,
                "removed": cert.removed,
                "p": cert.p,
                "gamma": 0.32,
                "lambda": np.nan,
                "alpha": 1.0 / 3.0,
                "eta": cert.eta,
                "accepted": 1.0 if cert.accepted_without_projection else 0.0,
            }
        )
    return out_rows


def write_latex_table(df: pd.DataFrame) -> None:
    labels = {
        "noisy": "Noisy",
        "fdncnn_color": "FDnCNN-Color",
        "drunet_color": "DRUNet-Color",
        "swinir_color": "SwinIR-Color",
        "ectv_fdncnn": "ECTV-Cert-FDnCNN",
        "ectv_drunet": "ECTV-Cert-DRUNet",
        "ectv_swinir": "ECTV-Cert-SwinIR",
    }
    method_order = ["noisy", "fdncnn_color", "drunet_color", "swinir_color", "ectv_fdncnn", "ectv_drunet", "ectv_swinir"]
    rows: list[list[str]] = []
    for method in method_order:
        group = df[df["method"] == method]
        if group.empty:
            continue
        eta = group["eta"].astype(float) if "eta" in group else pd.Series([], dtype=float)
        full_accept = eta >= 1.0 - 1e-7
        no_op = eta <= 1e-7
        projection = ~(full_accept | no_op)
        ratio = group["removed"] / group["budget"].replace(0, np.nan)
        rows.append(
            [
                labels.get(method, method),
                str(len(group)),
                fmt(float(group["psnr"].mean())),
                fmt(float(group["ssim"].mean()), 4),
                fmt(float(group["grad_corr"].mean()), 4),
                fmt(float(ratio.mean())) if method.startswith("ectv_") else "--",
                fmt(100.0 * float(full_accept.mean()), 1) if method.startswith("ectv_") else "--",
                fmt(100.0 * float(projection.mean()), 1) if method.startswith("ectv_") else "--",
                fmt(float(eta.median()), 4) if method.startswith("ectv_") else "--",
                fmt(float(group["runtime"].mean()), 2),
            ]
        )
    lines = [
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Method & Cases & PSNR & SSIM & Grad. corr. & Removed/Budget & Kept (\%) & Proj. (\%) & Med. $\eta$ & Runtime (s) \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "tab_transformer_baseline_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a SwinIR transformer denoising audit for the ECTV paper.")
    parser.add_argument("--kodak-count", type=int, default=24)
    parser.add_argument("--table-only", action="store_true", help="Regenerate the LaTeX table from results/transformer_baseline_audit.csv without rerunning SwinIR.")
    args = parser.parse_args()

    csv_path = RESULTS / "transformer_baseline_audit.csv"
    if args.table_only:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        write_latex_table(pd.read_csv(csv_path))
        print(f"Wrote {GENERATED / 'tab_transformer_baseline_audit.tex'}")
        return 0

    cases = select_kodak_cases(args.kodak_count)
    if not cases:
        raise RuntimeError("No Kodak cases selected for SwinIR audit.")

    model, torch = load_swinir()
    rows: list[dict[str, object]] = []
    rows.extend(append_existing_references(cases, RESULTS / "modern_baseline_subset.csv", ["noisy", "fdncnn_color", "drunet_color", "ectv_fdncnn", "ectv_drunet"]))
    rows.extend(run_cases(cases, model, torch))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_latex_table(df)
    manifest = {
        "model": "SwinIR color denoising",
        "weights": str(SWINIR_WEIGHTS.relative_to(ROOT)),
        "network_file": str((SWINIR_CODE / "network_swinir.py").relative_to(ROOT)),
        "kodak_count": len(cases),
        "sigma255": 25.0,
        "seed": 0,
        "command": f"python experiments/run_swinir_audit.py --kodak-count {args.kodak_count}",
        "note": "Official SwinIR color denoising noise25 fixed-slice audit on Kodak24.",
    }
    (RESULTS / "transformer_baseline_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {GENERATED / 'tab_transformer_baseline_audit.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
