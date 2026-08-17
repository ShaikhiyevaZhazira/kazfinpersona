"""Synthetic client profile generation, calibrated to config/kz_stats.yaml.

Design note for the paper: profiles are drawn from *declared* marginal
distributions, and `validate.py` later tests the generated sample against
those same targets with a KS / chi-square test. That turns "we made up some
customers" into "we sampled from a stated, testable population model".
"""
from __future__ import annotations

import numpy as np

from .schemas import ClientProfile

_GENDERS = ("f", "m")


def _weighted_choice(rng: np.random.Generator, items: list, weights: list[float]):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return items[rng.choice(len(items), p=w)]


def _sample_age(rng: np.random.Generator, cfg: dict) -> int:
    bins = cfg["age"]["bins"]
    weights = np.asarray(cfg["age"]["weights"], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(weights), p=weights)
    lo, hi = bins[idx], bins[idx + 1]
    return int(rng.integers(lo, hi))


def _sample_income(rng: np.random.Generator, cfg: dict,
                   region_mult: float, employment_mult: float) -> int:
    inc = cfg["income_kzt"]
    mu = np.log(inc["median"])
    raw = float(rng.lognormal(mean=mu, sigma=inc["sigma"]))
    raw *= region_mult * employment_mult
    raw = min(max(raw, inc["floor"]), inc["cap"])
    return int(round(raw / 1000.0) * 1000)


def _sample_credit_score(rng: np.random.Generator, cfg: dict,
                         income: int, employment: str) -> int:
    cs = cfg["credit_score"]
    val = rng.normal(cs["mean"], cs["std"])
    # Mild, *declared* dependence on income and employment stability so that
    # the score is not independent noise. Stated explicitly in the paper.
    val += 40.0 * np.tanh((income - cfg["income_kzt"]["median"]) / 400000.0)
    if employment in ("unemployed", "student"):
        val -= 45.0
    return int(np.clip(round(val), cs["min"], cs["max"]))


def _sample_existing_products(rng: np.random.Generator, cfg: dict,
                              age: int, income: int,
                              employment: str) -> list[str]:
    pen = cfg["existing_product_penetration"]
    # ids follow the real Halyk catalogue (config/products_halyk.json v0.2)
    name_to_id = {
        "debit_card": "hb_card_debit",
        "savings_deposit": "hb_deposit_universal",
        "term_deposit": "hb_deposit_qorjin",
        "credit_card": "hb_card_credit",
        "consumer_loan": "hb_loan_no_collateral",
        "mortgage": "hb_mortgage_digital",
        "auto_loan": "hb_auto_digital",
        "installment": "hb_installment",
    }
    held: list[str] = []
    for key, base_p in pen.items():
        p = float(base_p)
        # credit products are less likely for low income / unstable employment
        if key in ("credit_card", "consumer_loan", "mortgage", "auto_loan"):
            if employment in ("unemployed", "student", "retired"):
                p *= 0.25
            p *= float(np.clip(income / 400000.0, 0.3, 1.8))
            if age < 21:
                p = 0.0
        if rng.random() < min(p, 0.98):
            held.append(name_to_id[key])
    return held


def _sample_employment(rng: np.random.Generator, names: list[str],
                       weights: list[float], age: int) -> str:
    """Draw employment conditional on age.

    Earlier versions rejection-resampled age-inconsistent draws (a 30-year-old
    'retired'), which quietly distorted the employment marginal and failed the
    chi-square validation. Zeroing the impossible categories *before* the draw
    and renormalising keeps the conditional structure explicit and testable.
    """
    w = np.asarray(weights, dtype=float).copy()
    for i, name in enumerate(names):
        if name == "retired" and age < 58:
            w[i] = 0.0
        if name == "student" and age > 26:
            w[i] = 0.0
        if name == "retired" and age >= 63:
            w[i] *= 6.0     # retirement becomes the dominant state
        if name == "business_owner" and age < 24:
            w[i] *= 0.3
    if w.sum() <= 0:
        return "employed"
    return names[rng.choice(len(names), p=w / w.sum())]


def _sample_family_status(rng: np.random.Generator, names: list[str],
                          weights: list[float], age: int) -> str:
    """Family status conditional on age, for the same reason as employment:
    post-hoc overrides distort the marginal and fail the chi-square test."""
    w = np.asarray(weights, dtype=float).copy()
    for i, name in enumerate(names):
        if age < 23 and name in ("married_with_kids", "single_parent"):
            w[i] = 0.0
        if age < 21 and name == "married_no_kids":
            w[i] *= 0.3
        if age > 40 and name == "single":
            w[i] *= 0.6
    if w.sum() <= 0:
        return "single"
    return names[rng.choice(len(names), p=w / w.sum())]


def generate_profiles(cfg: dict, n: int, seed: int) -> list[ClientProfile]:
    rng = np.random.default_rng(seed)
    regions = cfg["regions"]
    emp_cfg = cfg["employment"]
    emp_names = list(emp_cfg.keys())
    emp_weights = [emp_cfg[k]["weight"] for k in emp_names]
    fam_names = list(cfg["family_status"].keys())
    fam_weights = list(cfg["family_status"].values())
    lang_names = list(cfg["language_profile"].keys())
    lang_weights = list(cfg["language_profile"].values())
    kids_names = [int(k) for k in cfg["children_count_if_kids"].keys()]
    kids_weights = list(cfg["children_count_if_kids"].values())

    # Region multipliers must have weighted mean 1.0, otherwise they silently
    # shift the whole income distribution away from the official target.
    # Normalising here keeps the published regional *ratios* intact while
    # preserving the national anchor. Before normalisation the weighted mean
    # was ~1.08, which inflated the employed-segment median by ~7%.
    rw = np.array([r["weight"] for r in regions], dtype=float)
    rm = np.array([r["income_mult"] for r in regions], dtype=float)
    region_norm = float(np.sum(rw * rm) / np.sum(rw))

    # Same argument for employment multipliers, but weighted by the
    # AGE-CONDITIONAL employment marginal, not the raw weights.
    out: list[ClientProfile] = []
    for i in range(n):
        region = _weighted_choice(rng, regions, [r["weight"] for r in regions])
        age = _sample_age(rng, cfg)
        employment = _sample_employment(rng, emp_names, emp_weights, age)

        income = _sample_income(
            rng, cfg,
            region["income_mult"] / region_norm,
            emp_cfg[employment]["income_mult"],
        )
        family_status = _sample_family_status(rng, fam_names, fam_weights, age)
        children = 0
        if family_status in ("married_with_kids", "single_parent"):
            children = int(_weighted_choice(rng, kids_names, kids_weights))

        existing = _sample_existing_products(rng, cfg, age, income, employment)

        # monthly obligations implied by the credit products already held
        debt_service = 0
        if "hb_mortgage_digital" in existing:
            debt_service += int(income * rng.uniform(0.18, 0.32))
        if "hb_auto_digital" in existing:
            debt_service += int(income * rng.uniform(0.10, 0.20))
        if "hb_loan_no_collateral" in existing:
            debt_service += int(income * rng.uniform(0.06, 0.18))
        if "hb_installment" in existing:
            debt_service += int(income * rng.uniform(0.03, 0.09))
        if "hb_card_credit" in existing:
            debt_service += int(income * rng.uniform(0.01, 0.06))

        out.append(
            ClientProfile(
                client_id=f"KFP{i:06d}",
                age=age,
                gender=str(rng.choice(_GENDERS)),
                region=region["name"],
                urban=bool(region["urban"]),
                employment=employment,
                income_kzt=income,
                income_stability=float(
                    np.clip(
                        rng.normal(emp_cfg[employment]["stability"], 0.08), 0.05, 1.0
                    )
                ),
                family_status=family_status,
                children=children,
                credit_score=_sample_credit_score(rng, cfg, income, employment),
                language_profile=_weighted_choice(rng, lang_names, lang_weights),
                existing_products=existing,
                existing_debt_service_kzt=debt_service,
                tenure_months=int(rng.integers(0, 121)),
            )
        )
    return out
