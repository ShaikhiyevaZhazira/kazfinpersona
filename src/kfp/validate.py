"""Statistical validation of the generated population.

Produces the appendix table that turns "synthetic data" from a weakness into
a controlled experimental design. For each marginal we state the target, the
realised value, and a goodness-of-fit test against the declared reference
distribution.

Note the honest framing to use in the paper: this validates that the
generator samples from the population model it *claims* to sample from. It
does not prove the population model matches Kazakhstan. That claim rests on
the cited official statistics, which is why every [VERIFY] in kz_stats.yaml
must be resolved with a real source.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import stats

from .schemas import ClientProfile


def validate_income(profiles: list[ClientProfile], cfg: dict) -> dict:
    """Validate income against the official anchor.

    Two distinct claims, tested separately -- conflating them was a bug:

    1. The `employed` segment must match the BNS figures, because that is what
       BNS actually measures (naemnye rabotniki). Tested with a KS test
       against Lognormal(ln(median), sigma) with the published parameters.

    2. The full population is a MIXTURE over employment types and regions, so
       it is not lognormal and a KS test against a lognormal would correctly
       reject it. Reported descriptively only. Saying otherwise in the paper
       would be a misuse of the test.
    """
    inc = np.array([p.income_kzt for p in profiles], dtype=float)
    emp = np.array([p.income_kzt for p in profiles
                    if p.employment == "employed"], dtype=float)
    target = cfg["income_kzt"]
    mu, sigma = np.log(target["median"]), target["sigma"]

    # A one-sample KS against a pure lognormal rejects even a correct sampler,
    # because the employed segment is itself a mixture over regions. With
    # n~4000 the test detects that structure at p<0.001 -- a true positive for
    # the wrong null hypothesis.
    #
    # Two claims, tested the way each should be:
    #   (a) sampler == declared model  -> two-sample KS vs a large MC draw
    #   (b) declared model == official -> relative error on median and mean,
    #       which is what BNS actually publishes.
    mc = _mc_income(cfg, n=300_000)
    ks = stats.ks_2samp(emp, mc) if emp.size else None
    rel_median = abs(np.median(emp) / target["median"] - 1.0) if emp.size else float("nan")
    rel_mean = (abs(np.mean(emp) / target["mean_target"] - 1.0)
                if emp.size and target.get("mean_target") else float("nan"))

    return {
        "metric": "income_kzt (employed segment)",
        "target_median": target["median"],
        "target_mean": target.get("mean_target"),
        "observed_median_employed": float(np.median(emp)) if emp.size else float("nan"),
        "observed_mean_employed": float(np.mean(emp)) if emp.size else float("nan"),
        "n_employed": int(emp.size),
        "ks_statistic": float(ks.statistic) if ks else float("nan"),
        "ks_pvalue": float(ks.pvalue) if ks else float("nan"),
        "rel_error_median": float(rel_median),
        "rel_error_mean": float(rel_mean),
        "population_median_all": float(np.median(inc)),
        "population_mean_all": float(np.mean(inc)),
        "population_p10": float(np.percentile(inc, 10)),
        "population_p90": float(np.percentile(inc, 90)),
        "gini": _gini(inc),
        "note": ("KS tested on the employed segment only; the full population is a "
                 "mixture over employment and region and is reported descriptively."),
    }


def _mc_income(cfg: dict, n: int = 300_000, seed: int = 12345) -> np.ndarray:
    """Monte-Carlo draw of employed-segment income from the declared model.

    Mirrors profiles._sample_income exactly, including region normalisation.
    Kept in sync by construction: if the sampler changes, this must change too,
    and the two-sample KS will fail loudly if it does not.
    """
    rng = np.random.default_rng(seed)
    regions = cfg["regions"]
    rw = np.array([r["weight"] for r in regions], dtype=float)
    rm = np.array([r["income_mult"] for r in regions], dtype=float)
    norm = float(np.sum(rw * rm) / np.sum(rw))
    inc = cfg["income_kzt"]
    mult = rng.choice(rm / norm, size=n, p=rw / rw.sum())
    emp_mult = cfg["employment"]["employed"]["income_mult"]
    raw = rng.lognormal(np.log(inc["median"]), inc["sigma"], n) * mult * emp_mult
    raw = np.clip(raw, inc["floor"], inc["cap"])
    return np.round(raw / 1000.0) * 1000.0


def _gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = x.size
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n)


def validate_categorical(profiles: list[ClientProfile], cfg: dict,
                         attr: str, target_map: dict) -> dict:
    obs = Counter(getattr(p, attr) for p in profiles)
    keys = list(target_map.keys())
    total = sum(obs.get(k, 0) for k in keys)
    if total == 0:
        return {"metric": attr, "error": "no observations"}
    observed = np.array([obs.get(k, 0) for k in keys], dtype=float)
    weights = np.array([
        v["weight"] if isinstance(v, dict) else v for v in target_map.values()
    ], dtype=float)
    expected = weights / weights.sum() * total
    chi2 = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "metric": attr,
        "categories": keys,
        "observed_share": (observed / total).round(4).tolist(),
        "target_share": (weights / weights.sum()).round(4).tolist(),
        "chi2_statistic": float(chi2.statistic),
        "chi2_pvalue": float(chi2.pvalue),
    }


def validate_age(profiles: list[ClientProfile], cfg: dict) -> dict:
    ages = np.array([p.age for p in profiles], dtype=float)
    bins = cfg["age"]["bins"]
    weights = np.array(cfg["age"]["weights"], dtype=float)
    weights = weights / weights.sum()
    observed, _ = np.histogram(ages, bins=bins)
    expected = weights * observed.sum()
    chi2 = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "metric": "age",
        "bins": bins,
        "observed_counts": observed.tolist(),
        "expected_counts": expected.round(1).tolist(),
        "chi2_statistic": float(chi2.statistic),
        "chi2_pvalue": float(chi2.pvalue),
        "observed_mean": float(ages.mean()),
    }


def validate_employment(profiles: list[ClientProfile], cfg: dict,
                        seed: int = 0, n_mc: int = 200_000) -> dict:
    """Employment is sampled *conditional on age*, so testing it against the
    raw target weights is the wrong null hypothesis — it would reject even a
    perfectly correct sampler.

    We therefore Monte-Carlo the marginal implied by the declared joint model
    (age distribution x age-conditional employment) and test against that,
    while still reporting the unconditional target for transparency.
    """
    from .profiles import _sample_age, _sample_employment  # local import

    rng = np.random.default_rng(seed)
    names = list(cfg["employment"].keys())
    weights = [cfg["employment"][k]["weight"] for k in names]

    counts = Counter()
    for _ in range(n_mc):
        age = _sample_age(rng, cfg)
        counts[_sample_employment(rng, names, weights, age)] += 1
    implied = np.array([counts.get(k, 0) for k in names], dtype=float)
    implied = implied / implied.sum()

    obs = Counter(p.employment for p in profiles)
    observed = np.array([obs.get(k, 0) for k in names], dtype=float)
    total = observed.sum()
    expected = implied * total
    chi2 = stats.chisquare(f_obs=observed, f_exp=expected)

    raw = np.array(weights, dtype=float)
    raw = raw / raw.sum()
    return {
        "metric": "employment",
        "categories": names,
        "observed_share": (observed / total).round(4).tolist(),
        "implied_share": implied.round(4).tolist(),
        "unconditional_target_share": raw.round(4).tolist(),
        "chi2_statistic": float(chi2.statistic),
        "chi2_pvalue": float(chi2.pvalue),
        "note": "tested against the age-conditional implied marginal",
    }


def validate_family_status(profiles: list[ClientProfile], cfg: dict,
                           seed: int = 0, n_mc: int = 200_000) -> dict:
    """Same age-conditional treatment as employment."""
    from .profiles import _sample_age, _sample_family_status

    rng = np.random.default_rng(seed)
    names = list(cfg["family_status"].keys())
    weights = list(cfg["family_status"].values())

    counts = Counter()
    for _ in range(n_mc):
        counts[_sample_family_status(rng, names, weights, _sample_age(rng, cfg))] += 1
    implied = np.array([counts.get(k, 0) for k in names], dtype=float)
    implied = implied / implied.sum()

    obs = Counter(p.family_status for p in profiles)
    observed = np.array([obs.get(k, 0) for k in names], dtype=float)
    total = observed.sum()
    chi2 = stats.chisquare(f_obs=observed, f_exp=implied * total)
    return {
        "metric": "family_status",
        "categories": names,
        "observed_share": (observed / total).round(4).tolist(),
        "implied_share": implied.round(4).tolist(),
        "chi2_statistic": float(chi2.statistic),
        "chi2_pvalue": float(chi2.pvalue),
        "note": "tested against the age-conditional implied marginal",
    }


def run_all(profiles: list[ClientProfile], cfg: dict) -> list[dict]:
    region_map = {r["name"]: r["weight"] for r in cfg["regions"]}
    return [
        validate_income(profiles, cfg),
        validate_age(profiles, cfg),
        validate_employment(profiles, cfg),
        validate_categorical(profiles, cfg, "region", region_map),
        validate_family_status(profiles, cfg),
        validate_categorical(profiles, cfg, "language_profile", cfg["language_profile"]),
    ]


def to_latex(rows: list[dict]) -> str:
    """Minimal booktabs table for the appendix."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Goodness-of-fit of the generated population against the declared reference distributions.}",
        r"\label{tab:dataset-validation}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Marginal & Statistic & $p$ \\",
        r"\midrule",
    ]
    for r in rows:
        if "ks_statistic" in r:
            lines.append(f"{r['metric']} (KS) & {r['ks_statistic']:.4f} & {r['ks_pvalue']:.3f} \\\\")
        elif "chi2_statistic" in r:
            lines.append(f"{r['metric']} ($\\chi^2$) & {r['chi2_statistic']:.2f} & {r['chi2_pvalue']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
