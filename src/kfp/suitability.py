"""Behavioural features + deterministic suitability ranking.

Eligibility answers "is this allowed?". Suitability answers "of the allowed
products, which one fits this client's behaviour and stated intent?".

Keeping the two separate is what lets you report the personalisation metric
and the compliance metric independently — and to show the trade-off curve
between them, which is the headline figure of the paper.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np

from .schemas import ClientProfile, Transaction

HALF_LIFE_DAYS = 90.0


def behavioural_features(
    profile: ClientProfile,
    txns: list[Transaction],
    as_of: dt.date,
) -> dict[str, float]:
    """Time-decayed behavioural summary.

    Recent behaviour is weighted more heavily via exponential decay with a
    90-day half-life. This is the feature vector used for segmentation and,
    later, for adapter routing.
    """
    if not txns:
        return {
            "n_txn": 0.0, "inflow": 0.0, "outflow": 0.0, "free_cashflow": 0.0,
            "savings_rate": 0.0, "spend_volatility": 0.0, "card_share": 0.0,
            "fx_share": 0.0, "education_share": 0.0, "transport_share": 0.0,
            "retail_share": 0.0, "debt_service_ratio": 0.0, "income_trend": 0.0,
        }

    lam = np.log(2.0) / HALF_LIFE_DAYS
    inflow = outflow = 0.0
    by_cat: dict[str, float] = defaultdict(float)
    by_channel: dict[str, float] = defaultdict(float)
    monthly_out: dict[str, float] = defaultdict(float)
    monthly_in: dict[str, float] = defaultdict(float)

    for t in txns:
        d = dt.date.fromisoformat(t.date)
        w = float(np.exp(-lam * max((as_of - d).days, 0)))
        amt = float(t.amount_kzt)
        month = t.date[:7]
        if amt > 0:
            inflow += amt * w
            monthly_in[month] += amt
        else:
            outflow += -amt * w
            by_cat[t.category] += -amt * w
            by_channel[t.channel] += -amt * w
            monthly_out[month] += -amt

    total_out = max(outflow, 1.0)
    months = sorted(monthly_out.keys())
    # Cold-start clients have a single month of history: std over one point is
    # undefined and propagated NaN into every downstream metric. Volatility is
    # simply unobservable there, and 0.0 with the cold-start flag already on the
    # scenario is the honest encoding.
    _out_vals = list(monthly_out.values())
    vol = (float(np.std(_out_vals) / max(np.mean(_out_vals), 1.0))
           if len(_out_vals) >= 2 else 0.0)

    trend = 0.0
    if len(months) >= 4:
        ins = np.array([monthly_in.get(m, 0.0) for m in months], dtype=float)
        x = np.arange(len(ins), dtype=float)
        if ins.mean() > 0:
            trend = float(np.polyfit(x, ins, 1)[0] / ins.mean())

    free_cf = (inflow - outflow) / max(len(months), 1)

    return {
        "n_txn": float(len(txns)),
        "inflow": inflow,
        "outflow": outflow,
        "free_cashflow": free_cf,
        "savings_rate": float(np.clip((inflow - outflow) / max(inflow, 1.0), -1.0, 1.0)),
        "spend_volatility": vol,
        "card_share": by_channel.get("card", 0.0) / total_out,
        "fx_share": by_cat.get("travel_fx", 0.0) / total_out,
        "education_share": by_cat.get("education", 0.0) / total_out,
        "transport_share": by_cat.get("transport", 0.0) / total_out,
        "retail_share": (by_cat.get("clothing", 0.0) + by_cat.get("entertainment", 0.0)) / total_out,
        "debt_service_ratio": by_cat.get("debt_service", 0.0) / total_out,
        "income_trend": trend,
    }


# ---------------------------------------------------------------- scoring

def _goal_match(product: dict, intent: str) -> float:
    return 1.0 if intent in product.get("goals", []) else 0.0


def suitability_score(
    product: dict,
    profile: ClientProfile,
    feats: dict[str, float],
    intent: str,
) -> float:
    """Transparent additive score. Weights are declared, not learned.

    Publish this function verbatim in the appendix: it is the definition of
    the gold label, so it must be inspectable and criticisable.
    """
    sig = product.get("suitability_signals", {})
    s = 0.0

    s += 3.0 * _goal_match(product, intent)

    min_cf = sig.get("min_free_cashflow_kzt")
    if min_cf is not None:
        s += 1.5 if feats["free_cashflow"] >= min_cf else -2.0

    if sig.get("prefers_stable_income"):
        s += 1.0 * (profile.income_stability - 0.5) * 2.0
    if sig.get("prefers_irregular_income"):
        s += 1.0 * (0.5 - profile.income_stability) * 2.0
    if sig.get("prefers_low_volatility_spending"):
        s += 1.0 * float(np.clip(1.0 - feats["spend_volatility"], -1.0, 1.0))
    if sig.get("prefers_high_card_usage"):
        s += 1.2 * (feats["card_share"] - 0.4)
    if sig.get("prefers_retail_spending"):
        s += 1.0 * (feats["retail_share"] - 0.10) * 4.0
    if sig.get("requires_fx_spending"):
        s += 2.0 if feats["fx_share"] > 0.02 else -2.5
    if sig.get("requires_transport_spending"):
        s += 1.0 if feats["transport_share"] > 0.08 else -1.0
    if sig.get("requires_existing_debt"):
        s += 2.0 if feats["debt_service_ratio"] > 0.05 else -3.0
    if sig.get("requires_existing_savings"):
        s += 1.5 if any(p.startswith("hb_deposit") for p in profile.existing_products) else -1.5
    if sig.get("prefers_older_client"):
        s += 1.0 * float(np.clip((profile.age - 35) / 20.0, -1.0, 1.0))

    # mild penalty for recommending a near-duplicate of something held
    if product["product_id"] in profile.existing_products:
        s -= 4.0

    return float(s)


def rank_eligible(
    catalog: dict,
    eligible: list[str],
    profile: ClientProfile,
    feats: dict[str, float],
    intent: str,
) -> list[tuple[str, float]]:
    by_id = {p["product_id"]: p for p in catalog["products"]}
    scored = [
        (pid, suitability_score(by_id[pid], profile, feats, intent))
        for pid in eligible
    ]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


def gold_decision(
    catalog: dict,
    eligible: list[str],
    profile: ClientProfile,
    feats: dict[str, float],
    intent: str,
    margin: float = 0.35,
) -> tuple[str | None, str]:
    """Return (gold_product_id, gold_action).

    A scenario becomes a *refusal* case when nothing eligible matches the
    intent — these are the negative cases the earlier dataset had none of,
    and without them compliance is not measurable.
    A scenario becomes *refer_to_advisor* when the top two candidates are
    within `margin`, i.e. the rules genuinely do not disambiguate. Forcing a
    single gold label there would inject label noise.
    """
    # Hard gate: a product may only be the gold answer if the client's intent
    # is among its declared goals. Without this gate the additive score can
    # push a topically unrelated product to the top (e.g. a credit card for a
    # vehicle-purchase intent), which would silently poison the gold labels.
    by_id = {p["product_id"]: p for p in catalog["products"]}
    on_goal = [pid for pid in eligible if intent in by_id[pid].get("goals", [])]

    ranked = rank_eligible(catalog, on_goal, profile, feats, intent)
    ranked = [(pid, sc) for pid, sc in ranked if sc > 0.0]
    if not ranked:
        return None, "refuse"
    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < margin:
        return ranked[0][0], "refer_to_advisor"
    return ranked[0][0], "recommend"
