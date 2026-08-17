"""Transaction history generation over an 18-month observation window.

Two things here that the previous pipeline lacked:

1. A real time axis. Transactions carry dates, so time-decayed behavioural
   features and preference *drift* become measurable.
2. Life events (job loss, new child, relocation, ...) that shift the
   spending regime mid-window. Drift is an open problem named in both
   RecGPT and INVESTMATE — you cannot study it without generating it.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .schemas import ClientProfile, Transaction, LifeEvent

_MERCHANT_BY_CATEGORY = {
    "food": "grocery",
    "housing_utils": "utility",
    "transport": "fuel",
    "communication": "telecom",
    "clothing": "retail",
    "health": "pharmacy",
    "education": "education",
    "entertainment": "leisure",
    "other": "misc",
}

_CHANNELS = ("card", "qr", "transfer", "cash", "autopay")
_CHANNEL_P = (0.52, 0.18, 0.14, 0.10, 0.06)


def _month_starts(end: dt.date, months: int) -> list[dt.date]:
    out = []
    y, m = end.year, end.month
    for _ in range(months):
        out.append(dt.date(y, m, 1))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return list(reversed(out))


def _sample_life_events(rng: np.random.Generator, cfg: dict,
                        client_id: str, months: list[dt.date]) -> list[LifeEvent]:
    events: list[LifeEvent] = []
    for name, p in cfg["life_events"].items():
        if rng.random() < float(p):
            # events land in the middle 60% of the window so that there is
            # both a pre- and a post-period to compare
            lo, hi = int(len(months) * 0.2), int(len(months) * 0.8)
            idx = int(rng.integers(lo, max(lo + 1, hi)))
            events.append(
                LifeEvent(
                    client_id=client_id,
                    date=months[idx].isoformat(),
                    event=name,
                    magnitude=float(np.clip(rng.normal(1.0, 0.25), 0.4, 2.0)),
                )
            )
    return events


def _apply_events(income: float, shares: dict[str, float],
                  active: list[str], magnitude: float) -> tuple[float, dict[str, float]]:
    shares = dict(shares)
    for ev in active:
        if ev == "job_loss":
            income *= 0.25 * magnitude
        elif ev == "income_shock_down":
            income *= 0.65
        elif ev == "salary_raise":
            income *= 1.0 + 0.18 * magnitude
        elif ev == "new_child":
            shares["health"] = shares.get("health", 0.04) * 1.9
            shares["food"] = shares.get("food", 0.44) * 1.12
            shares["entertainment"] = shares.get("entertainment", 0.04) * 0.55
        elif ev == "relocation":
            shares["housing_utils"] = shares.get("housing_utils", 0.13) * 1.5
            shares["transport"] = shares.get("transport", 0.09) * 1.3
    total = sum(shares.values())
    if total > 1.0:
        shares = {k: v / total * 0.97 for k, v in shares.items()}
    return income, shares


def generate_transactions(
    profile: ClientProfile,
    cfg: dict,
    seed: int,
    months: int = 18,
    end_date: dt.date | None = None,
    cold_start: bool = False,
) -> tuple[list[Transaction], list[LifeEvent]]:
    rng = np.random.default_rng(seed)
    end_date = end_date or dt.date(2026, 6, 1)
    month_list = _month_starts(end_date, months)

    if cold_start:
        month_list = month_list[-1:]

    events = _sample_life_events(rng, cfg, profile.client_id, month_list)
    event_by_month: dict[str, list[LifeEvent]] = {}
    for ev in events:
        event_by_month.setdefault(ev.date, []).append(ev)

    base_shares = dict(cfg["spending_shares"])
    conc = float(cfg["dirichlet_concentration"])
    keys = list(base_shares.keys())
    alpha = np.array([base_shares[k] * conc for k in keys], dtype=float)
    personal = rng.dirichlet(alpha)

    # A Dirichlet draw sums to EXACTLY 1.0, so using it directly consumed the
    # whole disposable income and left every client with zero savings capacity
    # by construction -- regardless of the fact that the configured shares sum
    # to 0.92. The Dirichlet governs the *composition* of spending; the overall
    # propensity to consume is a separate, per-client parameter.
    #
    # Beta(a, b) is scaled to [0.55, 1.02]: a minority of clients genuinely
    # overspend (ratio > 1, financed by credit), most save something. The mean
    # is set from the configured share total so the config stays meaningful.
    target_total = float(sum(cfg["spending_shares"].values()))
    lo, hi = 0.55, 1.02
    m = (target_total - lo) / (hi - lo)
    conc_pc = 12.0
    propensity = lo + (hi - lo) * float(rng.beta(m * conc_pc, (1 - m) * conc_pc))
    shares = {k: float(v) * propensity for k, v in zip(keys, personal)}

    txns: list[Transaction] = []
    active_events: list[str] = []
    magnitude = 1.0

    for mstart in month_list:
        for ev in event_by_month.get(mstart.isoformat(), []):
            active_events.append(ev.event)
            magnitude = ev.magnitude

        income = float(profile.income_kzt)
        income *= float(np.clip(rng.normal(1.0, 1.0 - profile.income_stability), 0.3, 1.9))
        income, month_shares = _apply_events(income, shares, active_events, magnitude)

        # Consumption shares apply to income NET of debt service, not gross.
        # Applying them to gross and then adding debt service on top pushed
        # every borrower into permanently negative cash flow (observed mean
        # free cash flow -18.7k, savings rate -0.14), which in turn made almost
        # no client suitable for a deposit. Debt is a prior claim on income;
        # households consume out of what is left.
        disposable = max(income - profile.existing_debt_service_kzt, income * 0.15)

        # salary inflow (1-2 per month for self-employed / business owners)
        n_inflows = 1 if profile.employment in ("employed", "retired") else int(rng.integers(1, 4))
        for k in range(n_inflows):
            day = int(rng.integers(1, 28))
            txns.append(
                Transaction(
                    client_id=profile.client_id,
                    date=mstart.replace(day=day).isoformat(),
                    category="income",
                    amount_kzt=int(income / n_inflows),
                    channel="transfer",
                    merchant_type="employer" if profile.employment == "employed" else "client",
                )
            )

        # outflows
        for cat, share in month_shares.items():
            budget = disposable * share
            if budget < 500:
                continue
            n_tx = int(np.clip(rng.poisson(_tx_rate(cat)), 1, 40))
            amounts = rng.dirichlet(np.ones(n_tx) * 2.2) * budget
            for amt in amounts:
                if amt < 200:
                    continue
                day = int(rng.integers(1, 29))
                txns.append(
                    Transaction(
                        client_id=profile.client_id,
                        date=mstart.replace(day=day).isoformat(),
                        category=cat,
                        amount_kzt=-int(amt),
                        channel=str(rng.choice(_CHANNELS, p=_CHANNEL_P)),
                        merchant_type=_MERCHANT_BY_CATEGORY.get(cat),
                    )
                )

        # debt service, if any
        if profile.existing_debt_service_kzt > 0:
            txns.append(
                Transaction(
                    client_id=profile.client_id,
                    date=mstart.replace(day=int(rng.integers(1, 12))).isoformat(),
                    category="debt_service",
                    amount_kzt=-profile.existing_debt_service_kzt,
                    channel="autopay",
                    merchant_type="bank",
                )
            )

        # occasional FX spending, used by the fx-deposit suitability signal
        if rng.random() < 0.06:
            txns.append(
                Transaction(
                    client_id=profile.client_id,
                    date=mstart.replace(day=int(rng.integers(1, 28))).isoformat(),
                    category="travel_fx",
                    amount_kzt=-int(rng.uniform(30000, 400000)),
                    channel="card",
                    merchant_type="foreign",
                    currency="USD",
                )
            )

    txns.sort(key=lambda t: t.date)
    return txns, events


def _tx_rate(category: str) -> float:
    return {
        "food": 18.0,
        "housing_utils": 3.0,
        "transport": 8.0,
        "communication": 2.0,
        "clothing": 2.5,
        "health": 2.0,
        "education": 1.5,
        "entertainment": 4.0,
        "other": 3.0,
    }.get(category, 3.0)
