from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ectv import delta_q, p_from_sigma

RESULTS = ROOT / "results"
GENERATED = ROOT / "generated"

DATASET_LABELS = {
    "synthetic_image": "Kodak24 synthetic",
    "real_camera": "PolyU real",
    "video_sequence": "Translated sequence",
}

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
}

CERT_TOL = 1e-7


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def write_rows(headers: list[str], rows: list[list[str]], path: Path, align: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if align is None:
        align = "l" + "c" * (len(headers) - 1)
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def fmt_ci(values: np.ndarray, higher_is_better: bool = True, digits: int = 3) -> tuple[str, str, str]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "--", "--", "--"
    rng = np.random.default_rng(20260522)
    boots = []
    for _ in range(2000):
        sample = values[rng.integers(0, values.size, values.size)]
        boots.append(float(np.mean(sample)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = float(np.mean(values))
    if not higher_is_better:
        mean, lo, hi = -mean, -hi, -lo
    return fmt(mean, digits), f"[{fmt(lo, digits)}, {fmt(hi, digits)}]", str(values.size)


def sign_flip_p_value(values: np.ndarray, trials: int = 50000) -> str:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "--"
    if np.allclose(values, 0.0):
        return "1.000"
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(20260524 + values.size)
    hits = 0
    done = 0
    batch = 5000
    while done < trials:
        size = min(batch, trials - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, values.size))
        means = np.mean(signs * values[None, :], axis=1)
        hits += int(np.sum(np.abs(means) >= observed - 1e-15))
        done += size
    p_value = (hits + 1.0) / (trials + 1.0)
    if p_value < 1e-3:
        return "$<10^{-3}$"
    return fmt(p_value, 3)


def paired_delta(df: pd.DataFrame, method_a: str, method_b: str, metric: str, keys: list[str]) -> np.ndarray:
    cols = keys + ["method", metric]
    sub = df[df["method"].isin([method_a, method_b])][cols].dropna()
    wide = sub.pivot_table(index=keys, columns="method", values=metric, aggfunc="mean")
    if method_a not in wide.columns or method_b not in wide.columns:
        return np.array([])
    return (wide[method_a] - wide[method_b]).dropna().to_numpy(dtype=float)


def load_result(name: str) -> pd.DataFrame:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def certificate_audit() -> None:
    rows: list[list[str]] = []
    for name in ["synthetic_image", "real_camera", "video_sequence"]:
        df = load_result(name)
        if df.empty or "accepted" not in df.columns:
            continue
        sub = df[df["method"] == "ectv_fdncnn"].copy()
        if sub.empty:
            continue
        ratio = sub["removed"] / sub["budget"].replace(0, np.nan)
        eta = sub["eta"].astype(float).fillna(1.0)
        full_accept = eta >= 1.0 - CERT_TOL
        no_op = eta <= CERT_TOL
        projection = ~(full_accept | no_op)
        compliance = sub["removed"].astype(float) <= sub["budget"].astype(float) + CERT_TOL
        eta_q25, eta_q75 = np.percentile(eta.to_numpy(dtype=float), [25, 75])
        rows.append(
            [
                DATASET_LABELS[name],
                str(len(sub)),
                fmt(100.0 * float(compliance.mean()), 1),
                fmt(100.0 * float(full_accept.mean()), 1),
                fmt(100.0 * float(projection.mean()), 1),
                fmt(100.0 * float(no_op.mean()), 1),
                fmt(float(sub["eta"].mean()), 4),
                fmt(float(sub["eta"].median()), 4),
                f"{fmt(float(eta_q25), 4)}/{fmt(float(eta_q75), 4)}",
                fmt(float(sub["budget"].mean())),
                fmt(float(sub["removed"].mean())),
                fmt(float(ratio.mean())),
                fmt(float(sub["p"].mean()), 4),
            ]
        )
    write_rows(
        [
            "Dataset",
            "Cases",
            "Cert. comp. (\\%)",
            "Kept (\\%)",
            "Proj. (\\%)",
            "No-op (\\%)",
            "Mean $\\eta$",
            "Med. $\\eta$",
            "$\\eta$ Q25/Q75",
            "Budget",
            "Removed",
            "Removed/Budget",
            "Mean $p$",
        ],
        rows,
        GENERATED / "tab_certificate_audit.tex",
        align="lcccccccccccc",
    )


def wrapper_value_summary() -> None:
    rows: list[list[str]] = []
    for name in ["synthetic_image", "real_camera", "video_sequence"]:
        df = load_result(name)
        if df.empty:
            continue
        cert = df[df["method"] == "ectv_fdncnn"].copy()
        prop = df[df["method"] == "fdncnn_color"].copy()
        noisy = df[df["method"] == "noisy"].copy()
        if cert.empty or prop.empty or noisy.empty:
            continue
        keys = ["task", "dataset", "image", "sigma255", "seed"]
        prop_metric = prop[keys + ["psnr", "ssim", "grad_corr"]].rename(columns={"psnr": "psnr_prop", "ssim": "ssim_prop", "grad_corr": "grad_prop"})
        noisy_metric = noisy[keys + ["psnr", "ssim"]].rename(columns={"psnr": "psnr_noisy", "ssim": "ssim_noisy"})
        merged = cert.merge(prop_metric, on=keys, how="inner").merge(noisy_metric, on=keys, how="inner")
        if merged.empty:
            continue
        ratio = merged["removed"] / merged["budget"].replace(0, np.nan)
        tv_ratio = merged["tv_output"] / merged["tv_input"].replace(0, np.nan)
        eta = merged["eta"].astype(float).fillna(1.0)
        full_accept = eta >= 1.0 - CERT_TOL
        no_op = eta <= CERT_TOL
        projection = ~(full_accept | no_op)
        rows.append(
            [
                DATASET_LABELS[name],
                str(len(merged)),
                fmt(100.0 * float(full_accept.mean()), 1),
                fmt(100.0 * float(projection.mean()), 1),
                fmt(100.0 * float(no_op.mean()), 1),
                fmt(float(merged["eta"].mean()), 4),
                fmt(float((1.0 - merged["eta"]).mean()), 4),
                fmt(float(ratio.mean())),
                fmt(float(tv_ratio.mean())),
                fmt(float((merged["psnr"] - merged["psnr_prop"]).mean())),
                fmt(float((merged["psnr"] - merged["psnr_noisy"]).mean())),
                fmt(float((merged["ssim"] - merged["ssim_prop"]).mean()), 4),
            ]
        )
    write_rows(
        [
            "Dataset",
            "Cases",
            "Kept (\\%)",
            "Proj. (\\%)",
            "No-op (\\%)",
            "Mean $\\eta$",
            "Mean $1-\\eta$",
            "Removed/Budget",
            "$\\TV(\\hat u)/\\TV(f)$",
            "$\\Delta$PSNR vs FDnCNN",
            "$\\Delta$PSNR vs Noisy",
            "$\\Delta$SSIM vs FDnCNN",
        ],
        rows,
        GENERATED / "tab_wrapper_value_summary.tex",
        align="lccccccccccc",
    )


def removed_variation_summary() -> None:
    rows: list[list[str]] = []
    for name in ["synthetic_image", "real_camera"]:
        df = load_result(name)
        if df.empty:
            continue
        for method in ["ectv", "ectv_fidelity", "ectv_fdncnn"]:
            sub = df[df["method"] == method].copy()
            if sub.empty:
                continue
            ratio = sub["removed"] / sub["budget"].replace(0, np.nan)
            preserved = sub["tv_output"] / sub["tv_input"].replace(0, np.nan)
            rows.append(
                [
                    DATASET_LABELS[name],
                    METHOD_LABELS[method],
                    fmt(float(sub["tv_input"].mean())),
                    fmt(float(sub["tv_output"].mean())),
                    fmt(float(sub["removed"].mean())),
                    fmt(float(sub["budget"].mean())),
                    fmt(float(ratio.mean())),
                    fmt(float(preserved.mean())),
                ]
            )
    write_rows(
        ["Dataset", "Method", "$\\TV(f)$", "$\\TV(\\hat u)$", "Removed", "Budget", "Removed/Budget", "$\\TV(\\hat u)/\\TV(f)$"],
        rows,
        GENERATED / "tab_removed_variation_summary.tex",
        align="llcccccc",
    )


def _candidate_removed(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "dataset", "image", "sigma255", "seed"]
    proposal = df[df["method"] == "fdncnn_color"][keys + ["tv_input", "tv_output"]].copy()
    proposal["candidate_removed"] = np.maximum(0.0, proposal["tv_input"] - proposal["tv_output"])
    cert = df[df["method"] == "ectv_fdncnn"][keys + ["alpha", "gamma"]].copy()
    return cert.merge(proposal[keys + ["candidate_removed"]], on=keys, how="inner")


def p_robustness_summary() -> None:
    rows: list[list[str]] = []
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
    for name in ["synthetic_image", "real_camera"]:
        df = load_result(name)
        if df.empty:
            continue
        merged = _candidate_removed(df)
        if merged.empty:
            continue
        sigma = merged["sigma255"].astype(float).to_numpy() / 255.0
        candidate_removed = merged["candidate_removed"].astype(float).to_numpy()
        alpha = merged["alpha"].astype(float).to_numpy()
        gamma = merged["gamma"].astype(float).to_numpy()
        for scale in scales:
            p_vals = np.array([p_from_sigma(s * scale) for s in sigma])
            budgets = np.array([delta_q(p, a, g) for p, a, g in zip(p_vals, alpha, gamma)])
            pass_rate = float(np.mean(candidate_removed <= budgets))
            rows.append(
                [
                    DATASET_LABELS[name],
                    fmt(scale, 2),
                    fmt(float(np.mean(p_vals)), 4),
                    fmt(float(np.mean(budgets))),
                    fmt(float(np.mean(candidate_removed / np.maximum(budgets, 1e-12)))),
                    fmt(100.0 * pass_rate, 1),
                ]
            )
    write_rows(
        ["Dataset", "$\\hat\\sigma$ scale", "Mean $p$", "Mean budget", "FDnCNN removed/Budget", "Proposal feasible (\\%)"],
        rows,
        GENERATED / "tab_p_robustness_summary.tex",
        align="lccccc",
    )


def budget_schedule_audit() -> None:
    rows: list[list[str]] = []
    p_min = 0.501
    for name in ["synthetic_image", "real_camera"]:
        df = load_result(name)
        if df.empty:
            continue
        merged = _candidate_removed(df)
        cert = df[df["method"] == "ectv_fdncnn"][["task", "dataset", "image", "sigma255", "seed", "p", "budget"]]
        keys = ["task", "dataset", "image", "sigma255", "seed"]
        merged = merged.merge(cert, on=keys, how="inner")
        if merged.empty:
            continue
        p_vals = merged["p"].astype(float).to_numpy()
        alpha = merged["alpha"].astype(float).to_numpy()
        gamma = merged["gamma"].astype(float).to_numpy()
        candidate_removed = merged["candidate_removed"].astype(float).to_numpy()
        current = merged["budget"].astype(float).to_numpy()
        schedules = {
            "ECTV tanh": current,
            "Linear $1-p$": np.clip(alpha * (1.0 - p_vals) / (1.0 - p_min), 0.0, alpha),
            "Entropy only": np.clip(alpha * binary_entropy(p_vals) / np.log(2.0), 0.0, alpha),
            "Dataset const.": np.full_like(current, float(np.nanmean(current))),
        }
        for label, budgets in schedules.items():
            budgets = np.asarray(budgets, dtype=float)
            pass_rate = np.mean(candidate_removed <= budgets)
            slack = budgets - candidate_removed
            rows.append(
                [
                    DATASET_LABELS[name],
                    label,
                    fmt(float(np.mean(budgets))),
                    fmt(float(np.median(budgets))),
                    fmt(100.0 * float(pass_rate), 1),
                    fmt(float(np.mean(candidate_removed / np.maximum(budgets, 1e-12)))),
                    fmt(float(np.mean(slack))),
                ]
            )
    write_rows(
        ["Dataset", "Schedule", "Mean budget", "Med. budget", "Proposal feasible (\\%)", "Removed/Budget", "Mean slack"],
        rows,
        GENERATED / "tab_budget_schedule_audit.tex",
        align="llccccc",
    )


def paired_delta_summary() -> None:
    comparisons: list[tuple[str, str, pd.DataFrame, list[str], str, str, str, bool]] = []
    synth = load_result("synthetic_image")
    real = load_result("real_camera")
    edge = load_result("downstream_edge")
    track = load_result("downstream_tracking")
    comp = load_result("downstream_compression")
    if not synth.empty:
        keys = ["dataset", "image", "sigma255", "seed"]
        comparisons.extend(
            [
                ("Kodak PSNR", "ECTV-Cert-FDnCNN vs FDnCNN", synth, keys, "ectv_fdncnn", "fdncnn_color", "psnr", True),
                ("Kodak PSNR", "ECTV-Fidelity vs TV-Discrepancy", synth, keys, "ectv_fidelity", "tv_discrepancy", "psnr", True),
                ("Kodak GradCorr", "ECTV-Tight vs TV-NoBudget", synth, keys, "ectv", "tv_maxiter", "grad_corr", True),
            ]
        )
    if not real.empty:
        keys = ["dataset", "image"]
        comparisons.extend(
            [
                ("PolyU PSNR", "ECTV-Cert-FDnCNN vs FDnCNN", real, keys, "ectv_fdncnn", "fdncnn_color", "psnr", True),
                ("PolyU PSNR", "ECTV-Fidelity vs TV-Discrepancy", real, keys, "ectv_fidelity", "tv_discrepancy", "psnr", True),
            ]
        )
    if not edge.empty:
        keys = ["dataset", "image", "sigma255"]
        comparisons.append(("Canny F1", "ECTV-Cert-FDnCNN vs Noisy", edge, keys, "ectv_fdncnn", "noisy", "edge_f1", True))
    if not track.empty:
        keys = ["dataset", "image", "sigma255", "transition"]
        comparisons.extend(
            [
                ("KLT EPE", "ECTV-Cert-FDnCNN vs Noisy", track, keys, "ectv_fdncnn", "noisy", "klt_epe", False),
                ("Farneback EPE", "ECTV-Cert-FDnCNN vs Noisy", track, keys, "ectv_fdncnn", "noisy", "farneback_epe", False),
            ]
        )
    if not comp.empty:
        keys = ["dataset", "image", "quality"]
        comparisons.append(("JPEG PSNR", "ECTV-Cert-FDnCNN vs Noisy", comp, keys, "ectv_fdncnn", "noisy", "psnr", True))

    rows: list[list[str]] = []
    metric_labels = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "grad_corr": "Grad. corr.",
        "edge_f1": "F1",
        "klt_epe": "KLT EPE",
        "farneback_epe": "Flow EPE",
    }
    for task, label, df, keys, a, b, metric, higher in comparisons:
        delta = paired_delta(df, a, b, metric, keys)
        if not higher:
            delta = -delta
        mean, ci, n = fmt_ci(delta, higher_is_better=True)
        if delta.size == 0 or np.allclose(delta, 0.0):
            win_rate = "--"
        else:
            win_rate = fmt(100.0 * float(np.mean(delta > 0)), 1)
        p_value = sign_flip_p_value(delta)
        rows.append([task, label, metric_labels.get(metric, metric.replace("_", "\\_")), n, mean, ci, win_rate, p_value])
    write_rows(
        ["Task", "Comparison", "Metric", "Pairs", "Mean signed $\\Delta$", "95\\% bootstrap CI", "Win (\\%)", "$p_{\\rm sf}$"],
        rows,
        GENERATED / "tab_paired_delta_summary.tex",
        align="llcccccc",
    )


def operating_guidance_table() -> None:
    rows = [
        ["High pass-through, $\\eta\\approx1$", "Proposal already satisfies the removed-TV budget.", "Admit the proposal and report the certificate as an audit tag."],
        ["Very low $\\eta$", "The budget conflicts with the proposal.", "Relax or task-calibrate the budget for fidelity-oriented restoration."],
        ["Dense motion prefers noisy input", "TV removal can erase cues used by KLT or flow.", "Select the operating point on the target motion operator."],
        ["Feature/edge/JPEG metrics improve", "Moderate denoising can help sparse geometry, contours, or coding.", "Use the certified operating point after task-level validation."],
        ["$\\Delta_Q\\ge\\TV(f)$", "The certificate becomes permissive.", "Report TV ratio and removed/budget with the certificate."],
    ]
    write_rows(
        ["Audit observation", "Interpretation", "Recommended action"],
        rows,
        GENERATED / "tab_operating_guidance.tex",
        align="p{0.25\\textwidth}p{0.30\\textwidth}p{0.36\\textwidth}",
    )


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", r"\_"))


def task_operating_region_table() -> None:
    candidate_methods = ["noisy", "ectv", "ectv_fidelity", "ectv_fdncnn"]
    rows: list[list[str]] = []

    def select_entries(series: pd.Series, higher_is_better: bool) -> tuple[str, str, str]:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            return "--", "--", "--"
        cand = vals[[m for m in candidate_methods if m in vals.index]].dropna()
        if cand.empty:
            selected_method = vals.idxmax() if higher_is_better else vals.idxmin()
        else:
            selected_method = cand.idxmax() if higher_is_better else cand.idxmin()
        best_method = vals.idxmax() if higher_is_better else vals.idxmin()
        selected = f"{method_label(str(selected_method))} ({fmt(float(vals.loc[selected_method]))})"
        best = f"{method_label(str(best_method))} ({fmt(float(vals.loc[best_method]))})"
        return str(selected_method), selected, best

    def add_row(pipeline: str, metric: str, series: pd.Series, higher_is_better: bool, action: str) -> None:
        selected_method, selected, best = select_entries(series, higher_is_better)
        if selected_method == "noisy":
            action_text = "Bypass denoising or recalibrate for this operator."
        else:
            action_text = action
        rows.append([pipeline, metric, selected, best, action_text])

    feature = load_result("downstream_feature_matching")
    if not feature.empty:
        grouped = feature.groupby(["detector", "method"])[["inliers", "corner_error"]].mean()
        if "orb" in grouped.index.get_level_values(0):
            orb = grouped.xs("orb")
            add_row("ORB registration", "Corner err. $\\downarrow$", orb["corner_error"], False, "Use certified learned output for sparse geometry.")
        if "sift" in grouped.index.get_level_values(0):
            sift = grouped.xs("sift")
            add_row("SIFT registration", "Corner err. $\\downarrow$", sift["corner_error"], False, "Use certified learned output for sparse geometry.")

    edge = load_result("downstream_edge")
    if not edge.empty:
        grouped = edge.groupby("method")[["edge_f1", "edge_recall"]].mean()
        add_row("Canny edges", "F1-1px $\\uparrow$", grouped["edge_f1"], True, "Use certified learned output for contour precision/recall balance.")

    tracking = load_result("downstream_tracking")
    if not tracking.empty:
        grouped = tracking.groupby("method")[["klt_epe", "farneback_epe"]].mean()
        add_row("KLT tracking", "EPE $\\downarrow$", grouped["klt_epe"], False, "Select the operating point on target motion validation.")
        add_row("Farneback flow", "EPE $\\downarrow$", grouped["farneback_epe"], False, "Select the operating point on target motion validation.")

    comp = load_result("downstream_compression")
    if not comp.empty:
        grouped = comp.groupby("method")[["psnr", "ssim"]].mean()
        add_row("JPEG compression", "Decoded PSNR $\\uparrow$", grouped["psnr"], True, "Use certified learned output before coding.")

    write_rows(
        ["Pipeline", "Calibration metric", "Task-selected point", "Best observed", "Deployment action"],
        rows,
        GENERATED / "tab_task_operating_regions.tex",
        align=r"lp{0.17\textwidth}p{0.19\textwidth}p{0.18\textwidth}p{0.28\textwidth}",
    )


def task_calibration_table() -> None:
    candidate_methods = ["noisy", "ectv", "ectv_fidelity", "ectv_fdncnn"]
    rows: list[list[str]] = []

    def choose(series: pd.Series, higher_is_better: bool) -> str:
        vals = series.dropna()
        vals = vals[[m for m in candidate_methods if m in vals.index]].dropna()
        if vals.empty:
            return ""
        return str(vals.idxmax() if higher_is_better else vals.idxmin())

    def summarize_selection(selected: list[str]) -> str:
        if not selected:
            return "--"
        counts: dict[str, int] = {}
        for method in selected:
            counts[method] = counts.get(method, 0) + 1
        parts = [f"{method_label(method)}:{count}" for method, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        return ", ".join(parts)

    def add_calibration_row(
        pipeline: str,
        metric_label: str,
        grouped: pd.DataFrame,
        metric: str,
        higher_is_better: bool,
    ) -> None:
        if grouped.empty:
            return
        images = sorted(str(image) for image in grouped["image"].dropna().unique())
        selected_methods: list[str] = []
        selector_values: list[float] = []
        noisy_values: list[float] = []
        cert_values: list[float] = []
        oracle_values: list[float] = []
        for image in images:
            train = grouped[grouped["image"].astype(str) != image]
            test = grouped[grouped["image"].astype(str) == image].set_index("method")
            if train.empty or test.empty:
                continue
            train_scores = train.groupby("method")[metric].mean()
            selected = choose(train_scores, higher_is_better)
            if not selected or selected not in test.index:
                continue
            test_scores = test[metric]
            oracle = choose(test_scores, higher_is_better)
            if not oracle:
                continue
            selected_methods.append(selected)
            selector_values.append(float(test_scores.loc[selected]))
            oracle_values.append(float(test_scores.loc[oracle]))
            if "noisy" in test.index:
                noisy_values.append(float(test_scores.loc["noisy"]))
            if "ectv_fdncnn" in test.index:
                cert_values.append(float(test_scores.loc["ectv_fdncnn"]))
        if not selector_values:
            return
        rows.append(
            [
                pipeline,
                metric_label,
                str(len(selector_values)),
                summarize_selection(selected_methods),
                fmt(float(np.mean(selector_values))),
                fmt(float(np.mean(noisy_values))) if noisy_values else "--",
                fmt(float(np.mean(cert_values))) if cert_values else "--",
                fmt(float(np.mean(oracle_values))),
            ]
        )

    feature = load_result("downstream_feature_matching")
    if not feature.empty:
        for detector, label in [("orb", "ORB registration"), ("sift", "SIFT registration")]:
            sub = feature[feature["detector"] == detector]
            grouped = sub.groupby(["image", "method"], as_index=False)["corner_error"].mean()
            add_calibration_row(label, "Corner err. $\\downarrow$", grouped, "corner_error", False)

    edge = load_result("downstream_edge")
    if not edge.empty:
        grouped = edge.groupby(["image", "method"], as_index=False)["edge_f1"].mean()
        add_calibration_row("Canny edges", "F1-1px $\\uparrow$", grouped, "edge_f1", True)

    tracking = load_result("downstream_tracking")
    if not tracking.empty:
        grouped = tracking.groupby(["image", "method"], as_index=False)[["klt_epe", "farneback_epe"]].mean()
        add_calibration_row("KLT tracking", "EPE $\\downarrow$", grouped, "klt_epe", False)
        add_calibration_row("Farneback flow", "EPE $\\downarrow$", grouped, "farneback_epe", False)

    comp = load_result("downstream_compression")
    if not comp.empty:
        grouped = comp.groupby(["image", "method"], as_index=False)["psnr"].mean()
        add_calibration_row("JPEG compression", "Decoded PSNR $\\uparrow$", grouped, "psnr", True)

    write_rows(
        ["Pipeline", "Metric", "Folds", "LOO selected methods", "Held-out selector", "Noisy", "Cert-FDnCNN", "Oracle"],
        rows,
        GENERATED / "tab_task_calibration_summary.tex",
        align=r"lp{0.14\textwidth}cp{0.25\textwidth}cccc",
    )


def main() -> None:
    certificate_audit()
    wrapper_value_summary()
    removed_variation_summary()
    budget_schedule_audit()
    paired_delta_summary()
    operating_guidance_table()
    task_operating_region_table()
    task_calibration_table()
    p_robustness_summary()
    print("Wrote revision audit tables to", GENERATED)


if __name__ == "__main__":
    main()
