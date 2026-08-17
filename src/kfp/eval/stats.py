"""Statistics and reporting.

The rejected version reported point estimates from a single seed and called a
0.0013 difference "the highest score". Everything here exists to make that
mistake structurally impossible.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values: list[float], statistic=np.mean,
                 n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point, lo, hi)."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = statistic(arr[idx], axis=1)
    return (
        float(statistic(arr)),
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def paired_test(a: list[float], b: list[float], test: str = "wilcoxon") -> dict:
    """Paired comparison of two systems on the same items.

    Wilcoxon by default: metric-level per-item scores are bounded and usually
    non-normal, so the t-test's assumptions do not hold. Report the effect
    size, not only the p-value.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
    a_arr, b_arr = a_arr[mask], b_arr[mask]
    if a_arr.size < 5 or np.allclose(a_arr, b_arr):
        return {"test": test, "statistic": float("nan"), "pvalue": float("nan"),
                "effect_size": 0.0, "n": int(a_arr.size)}
    if test == "wilcoxon":
        res = stats.wilcoxon(a_arr, b_arr)
    else:
        res = stats.ttest_rel(a_arr, b_arr)
    diff = a_arr - b_arr
    pooled = np.std(diff, ddof=1)
    return {
        "test": test,
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "effect_size": float(np.mean(diff) / pooled) if pooled > 0 else 0.0,
        "mean_diff": float(np.mean(diff)),
        "n": int(a_arr.size),
    }


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Multiple-comparison correction.

    An ablation table with eight comparisons at alpha=0.05 will produce a
    spurious 'significant' result about a third of the time. Correct it, and
    say in the paper that you did.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        adj = max(prev, min(1.0, p * (m - i)))
        prev = adj
        out[name] = {"p_raw": p, "p_adjusted": adj, "threshold": thresh,
                     "significant": adj < alpha}
    return out


def aggregate_seeds(runs: list[dict[str, float]]) -> dict[str, dict]:
    """mean +/- std across seeds for every metric. Never report one seed."""
    keys = set().union(*[set(r.keys()) for r in runs]) if runs else set()
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in runs if k in r and not np.isnan(r[k])]
        if not vals:
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "values": [float(v) for v in vals],
        }
    return out


# Direction of improvement per metric. Bolding the maximum of a
# lower-is-better metric would highlight the WORST system — a silent and
# very embarrassing table bug. Anything absent here is treated as diagnostic
# and never bolded, because "higher is better" is not always true:
# answer_diversity, for instance, is HIGHER for a noisy system than for the
# oracle, so it must be read alongside the counterfactual metrics, never
# ranked on its own.
HIGHER_IS_BETTER = {
    "action_accuracy", "product_match_accuracy", "profile_sensitivity",
    "counterfactual_correctness", "refusal_precision", "refusal_recall",
    "refusal_f1", "fact_coverage",
}
LOWER_IS_BETTER = {
    "constraint_violation_rate", "hard_violation_rate", "unparseable_rate",
    "fact_fabrication_rate", "sensitivity_gap", "fact_hallucination_rate",
}


def results_to_latex(table: dict[str, dict[str, dict]],
                     metrics: list[str],
                     caption: str = "Main results.",
                     label: str = "tab:main") -> str:
    """table[system][metric] = {'mean':..,'std':..} -> booktabs LaTeX.

    Metrics carry an arrow in the header so the reader never has to guess the
    direction, and only metrics with a defined direction are bolded.
    """
    def _hdr(m):
        arrow = r"$\uparrow$" if m in HIGHER_IS_BETTER else (
            r"$\downarrow$" if m in LOWER_IS_BETTER else "")
        return m.replace("_", r"\_") + (" " + arrow if arrow else "")

    header = " & ".join(["System"] + [_hdr(m) for m in metrics])
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "r" * len(metrics) + "}",
        r"\toprule", header + r" \\", r"\midrule",
    ]
    best = {}
    for m in metrics:
        vals = [(s, table[s].get(m, {}).get("mean", float("nan"))) for s in table]
        vals = [(s, v) for s, v in vals if not np.isnan(v)]
        if not vals:
            continue
        if m in HIGHER_IS_BETTER:
            best[m] = max(vals, key=lambda kv: kv[1])[0]
        elif m in LOWER_IS_BETTER:
            best[m] = min(vals, key=lambda kv: kv[1])[0]
        # diagnostic metrics: no winner, nothing bolded
    for system, row in table.items():
        cells = []
        for m in metrics:
            d = row.get(m)
            if d is None or np.isnan(d.get("mean", float("nan"))):
                cells.append("--")
                continue
            txt = f"{d['mean']:.3f}"
            if d.get("std"):
                txt += rf"$\pm${d['std']:.3f}"
            if best.get(m) == system:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join([system.replace("_", r"\_")] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)
