"""Metric definitions for KazFinPersona.

The set is organised around one claim: **accuracy and fluency do not measure
personalization**. Each family below isolates a different failure mode that
semantic similarity is blind to.

    accuracy family      -> is the answer right?
    personalization      -> does the answer depend on the client at all?
    compliance           -> is the answer permissible?
    grounding            -> are the asserted facts real?
    robustness           -> does it survive a language switch?
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# ===================================================================
# accuracy family
# ===================================================================

def action_accuracy(records: list[dict]) -> float:
    """3-class accuracy over recommend / refuse / refer_to_advisor."""
    if not records:
        return float("nan")
    return float(np.mean([r["pred_action"] == r["gold_action"] for r in records]))


def product_match_accuracy(records: list[dict]) -> float:
    """Correct product, computed only where the gold action is `recommend`.

    Reported alongside `recommend_coverage` so a model cannot inflate PMA by
    declining most items.
    """
    sub = [r for r in records if r["gold_action"] == "recommend"]
    if not sub:
        return float("nan")
    return float(np.mean([r["pred_product_id"] == r["gold_product_id"] for r in sub]))


def refusal_metrics(records: list[dict]) -> dict:
    """How reliably does the model decline when declining is correct?

    A financial assistant that never says no is unusable regardless of its
    accuracy on the positive cases.
    """
    gold_ref = [r["gold_action"] == "refuse" for r in records]
    pred_ref = [r["pred_action"] == "refuse" for r in records]
    tp = sum(g and p for g, p in zip(gold_ref, pred_ref))
    fp = sum((not g) and p for g, p in zip(gold_ref, pred_ref))
    fn = sum(g and (not p) for g, p in zip(gold_ref, pred_ref))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    # NaN-safe: recall 0 with defined precision must give F1 = 0, not NaN.
    # (`prec and rec` was falsy for 0.0 and silently turned F1 into NaN.)
    if np.isnan(rec):
        f1 = float("nan")                    # no gold refusals: undefined
    elif rec == 0 or (not np.isnan(prec) and prec + rec == 0):
        f1 = 0.0                             # model never refuses correctly
    elif np.isnan(prec):
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"refusal_precision": prec, "refusal_recall": rec, "refusal_f1": f1}


def unparseable_rate(records: list[dict]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([r.get("unparseable", False) for r in records]))


# ===================================================================
# personalization family  -- the core contribution
# ===================================================================

def profile_sensitivity(pairs: list[tuple[dict, dict]]) -> float:
    """Share of counterfactual pairs where the prediction changes AT ALL.

    Near zero means the model is answering from the question alone and
    ignoring the client profile — the failure mode that high semantic
    similarity conceals.
    """
    if not pairs:
        return float("nan")
    changed = [
        (a["pred_action"], a["pred_product_id"]) != (b["pred_action"], b["pred_product_id"])
        for a, b in pairs
    ]
    return float(np.mean(changed))


def counterfactual_correctness(pairs: list[tuple[dict, dict]]) -> float:
    """Share of pairs where the prediction changes *to the correct new gold*.

    Every pair here is one where the rule engine's decision flips, so a model
    that tracks the rules must flip too, and in the right direction.
    """
    if not pairs:
        return float("nan")
    ok = []
    for base, cf in pairs:
        ok.append(
            cf["pred_action"] == cf["gold_action"]
            and cf["pred_product_id"] == cf["gold_product_id"]
            and (cf["pred_action"], cf["pred_product_id"])
            != (base["pred_action"], base["pred_product_id"])
        )
    return float(np.mean(ok))


def sensitivity_correctness_gap(pairs: list[tuple[dict, dict]]) -> float:
    """PS - CC. Large gap = the model reacts to the profile, but wrongly.

    This is a diagnostic, not a score: a model can be `insensitive`
    (both near 0) or `miscalibrated` (PS high, CC low), and the remedies
    differ. Report both numbers, never their ratio alone.
    """
    return profile_sensitivity(pairs) - counterfactual_correctness(pairs)


def answer_diversity(records: list[dict]) -> float:
    """Normalised entropy of predicted products within each intent.

    A complementary, pair-free view of the same phenomenon: if the model emits
    one product per intent regardless of client, entropy collapses to 0.
    """
    by_intent: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_intent[r["intent"]].append(str(r["pred_product_id"]))
    ents = []
    for _, preds in by_intent.items():
        if len(preds) < 2:
            continue
        _, counts = np.unique(preds, return_counts=True)
        p = counts / counts.sum()
        h = -np.sum(p * np.log(p + 1e-12))
        hmax = np.log(len(preds))
        if hmax > 0:
            ents.append(h / hmax)
    return float(max(0.0, np.mean(ents))) if ents else float("nan")


# ===================================================================
# compliance family
# ===================================================================

def constraint_violation_rate(records: list[dict]) -> float:
    """Share of recommendations naming a product the client is NOT eligible for.

    Denominator is the set of responses that actually recommend something.
    This is the metric that makes the domain regulated rather than generic:
    an answer can be fluent, on-topic and semantically close to the reference
    while being one the bank is not permitted to make.
    """
    sub = [r for r in records if r["pred_action"] in ("recommend", "refer_to_advisor")
           and r["pred_product_id"]]
    if not sub:
        return float("nan")
    return float(np.mean([r["pred_product_id"] not in r["eligible_products"] for r in sub]))


def violation_by_rule(records: list[dict], verdict_lookup) -> dict:
    """Which rule gets violated most often. Feeds the error-analysis table."""
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        pid = r.get("pred_product_id")
        if not pid or r["pred_action"] == "refuse":
            continue
        if pid in r["eligible_products"]:
            continue
        for rule in verdict_lookup(r["client_id"], pid):
            counts[rule] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# Non-negotiable rules. The two age variants were missing before v1.0: an
# age ceiling that depends on gender or on the loan maturity is as
# non-negotiable as a plain age ceiling.
HARD_RULES = {"age_min", "age_max", "age_max_by_gender", "age_max_at_maturity",
              "employment", "requires_children", "excludes_if_holds",
              "requires_holds"}


def hard_violation_rate(records: list[dict]) -> float:
    """Violations of *non-negotiable* rules only (age, employment, holdings).

    Separated from soft thresholds like DTI because a reviewer will rightly
    argue that a borderline DTI is a judgement call while recommending a
    mortgage to a 17-year-old is not.
    """
    HARD = HARD_RULES
    sub = [r for r in records if r["pred_action"] in ("recommend", "refer_to_advisor")
           and r["pred_product_id"]]
    if not sub:
        return float("nan")
    flags = []
    for r in sub:
        rules = set()
        for item in r.get("ineligible_products", []):
            if item["product_id"] == r["pred_product_id"]:
                rules = set(item["violated_rules"])
        flags.append(bool(rules & HARD))
    return float(np.mean(flags))


# ===================================================================
# grounding family
# ===================================================================

def fact_grounding(records: list[dict], catalog: dict,
                   parser=None) -> dict:
    """Fact-level grounding, split by what the answer was supposed to contain.

    Two defects were found by running the ORACLE through this metric, which by
    construction cannot be wrong. Both were in the metric, not the model.

    1. `fact_coverage` counted refusal scenarios in its denominator. A refusal
       carries no `product_id=` fact -- there is no product to name -- so 281
       correct refusals were scored as 0% coverage and dragged the oracle down
       to 0.528. Coverage is now computed only over items where a product was
       supposed to be named, and refusals get their own metric.

    2. `fact_fabrication_rate` flagged the oracle at 3.2%. The culprit was the
       product NAME: "7-20-25" yields the numbers 7, 20 and 25. Seven is the
       rate and twenty is the down payment, both in the catalogue -- but 25 is
       the term in YEARS, while the catalogue stores 300 MONTHS. The model
       invented nothing. Numbers are now checked against unit-aware variants,
       and digits belonging to the product name are excluded outright.

    Returned:
      fact_coverage           did the answer name the required product?
                              (denominator: recommend / refer_to_advisor only)
      refusal_reason_coverage did the refusal cite a blocking rule?
                              (denominator: refuse only)
      fact_fabrication_rate   numeric claims absent from the catalogue entry
    """
    by_id = {p["product_id"]: p for p in catalog["products"]}
    cov, fab, ref_cov = [], [], []

    for r in records:
        pid = r.get("pred_product_id")
        gold_action = r.get("gold_action")

        # ---- coverage, only where a product was required
        req_products = [f for f in r.get("rationale_facts", [])
                        if f.startswith("product_id=")]
        if req_products:
            want = req_products[0].split("=", 1)[1]
            cov.append(1.0 if pid == want else 0.0)

        # ---- refusal reason coverage
        if gold_action == "refuse":
            text = (r.get("answer_text") or "").lower()
            rule = ""
            for f in r.get("rationale_facts", []):
                if f.startswith("primary_blocking_rule="):
                    rule = f.split("=", 1)[1]
            ref_cov.append(1.0 if (rule and rule.lower() in text) else 0.0)

        # ---- fabrication
        if pid and pid in by_id:
            prod = by_id[pid]
            allowed = _allowed_numbers(prod)
            name_digits = _numbers_in_text(
                f"{prod.get('name_ru','')} {prod.get('name_kk','')} {pid}")
            claims = {n for n in (round(float(x), 2)
                                  for x in r.get("asserted_numbers", []))
                      if n >= 1.0}
            claims -= name_digits          # digits from the product's own name
            if claims:
                fab.append(len(claims - allowed) / len(claims))

    return {
        "fact_coverage": float(np.mean(cov)) if cov else float("nan"),
        "refusal_reason_coverage": float(np.mean(ref_cov)) if ref_cov else float("nan"),
        "fact_fabrication_rate": float(np.mean(fab)) if fab else float("nan"),
    }


def _numbers_in_text(text: str) -> set[float]:
    import re
    out = set()
    for m in re.finditer(r"\d+(?:[.,]\d+)?", text or ""):
        try:
            out.add(round(float(m.group(0).replace(",", ".")), 2))
        except ValueError:
            pass
    return out


def _allowed_numbers(product: dict) -> set[float]:
    """Every number a truthful answer about this product could contain.

    Unit-aware: a term stored as 300 months is legitimately spoken as 25 years,
    and a rate of 14.93% may be rounded to 14.9 or 15. Without these variants
    the metric punishes correct, natural phrasing as hallucination.
    """
    allowed: set[float] = set()

    def add(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        f = round(float(v), 2)
        allowed.add(f)
        allowed.add(round(f, 1))
        allowed.add(float(round(f)))
        if f >= 12 and abs(f / 12 - round(f / 12)) < 1e-6:
            allowed.add(float(round(f / 12)))      # months -> years
        if f >= 1000:
            allowed.add(round(f / 1000, 2))        # tenge -> thousands
            allowed.add(round(f / 1_000_000, 2))   # tenge -> millions

    def walk(v):
        if isinstance(v, dict):
            for k, sub in v.items():
                try:
                    add(float(k))                  # rate_by_term keys are terms
                except (TypeError, ValueError):
                    pass
                walk(sub)
        elif isinstance(v, list):
            for sub in v:
                walk(sub)
        else:
            add(v)

    walk(product.get("attributes", {}))
    walk(product.get("eligibility", {}))
    return allowed


# ===================================================================
# robustness family
# ===================================================================

def code_switching_degradation(by_condition: dict[str, list[dict]],
                               metric_fn) -> dict:
    """Delta of any metric across kk / ru / code-switched conditions.

    Deployed Kazakh assistants operate bilingually and users mix languages
    freely, yet no existing benchmark measures what that does to a financial
    recommendation. Report absolute values per condition plus the drop.
    """
    scores = {cond: metric_fn(recs) for cond, recs in by_condition.items()}
    base = scores.get("kk", float("nan"))
    out = dict(scores)
    for cond, val in scores.items():
        if cond != "kk":
            out[f"delta_{cond}_vs_kk"] = val - base
    return out

def profile_sensitivity_parsed(pairs):
    """PS только по парам, где разобраны ОБА ответа. Переход «разобран ->
    неразобран» не является реакцией на профиль, это шум парсинга."""
    ok = [(a, b) for a, b in pairs
          if a["pred_action"] is not None and b["pred_action"] is not None]
    if not ok:
        return float("nan")
    return float(np.mean([
        (a["pred_action"], a["pred_product_id"]) != (b["pred_action"], b["pred_product_id"])
        for a, b in ok]))


# ===================================================================
# v1.0 additions
# ===================================================================

def pair_type(base: dict, cf: dict) -> str:
    """Transition label of a counterfactual pair, e.g. 'recommend->refuse'."""
    return f"{base['gold_action']}->{cf['gold_action']}"


def pair_breakdown(pairs: list[tuple[dict, dict]]) -> dict:
    """PS and CC per gold-transition type.

    The pooled PS/CC hide that most pairs are recommend->recommend, where the
    only way to be right is to switch to a different product. Report n with
    every cell: several types have fewer than 5 pairs and must not be read as
    rates.
    """
    groups: dict[str, list] = defaultdict(list)
    for a, b in pairs:
        groups[pair_type(a, b)].append((a, b))
    return {k: {"n": len(v),
                "profile_sensitivity": profile_sensitivity(v),
                "counterfactual_correctness": counterfactual_correctness(v)}
            for k, v in sorted(groups.items())}


def multi_product_rate(records: list[dict]) -> float:
    """Share of parsed answers that recommend MORE THAN ONE product.

    The parser takes the first non-negated mention as the prediction, so a
    list answer ("1. hb_deposit_maximum 2. hb_deposit_qorjin ...") is scored
    on its first item. PMA on such answers rewards listing in catalogue order.
    Needs `recommended_ids` on the records (set by re-parsing answer_text).
    """
    sub = [r for r in records
           if not r.get("unparseable") and r.get("recommended_ids") is not None]
    if not sub:
        return float("nan")
    return float(np.mean([len(set(r["recommended_ids"])) > 1 for r in sub]))


def product_match_single(records: list[dict]) -> float:
    """Strict PMA: only gold-recommend items whose answer names exactly one
    recommended product. A model hedging with a list gets no credit."""
    sub = [r for r in records if r["gold_action"] == "recommend"
           and r.get("recommended_ids") is not None]
    if not sub:
        return float("nan")
    return float(np.mean([
        len(set(r["recommended_ids"])) == 1
        and r["pred_product_id"] == r["gold_product_id"] for r in sub]))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def counts(records: list[dict], pairs: list[tuple[dict, dict]]) -> dict:
    """Denominators for every headline metric, so a table can print n."""
    return {
        "n_base": len(records),
        "n_gold_recommend": sum(r["gold_action"] == "recommend" for r in records),
        "n_gold_refuse": sum(r["gold_action"] == "refuse" for r in records),
        "n_gold_refer": sum(r["gold_action"] == "refer_to_advisor" for r in records),
        "n_recommending": sum(r["pred_action"] in ("recommend", "refer_to_advisor")
                              and bool(r["pred_product_id"]) for r in records),
        "n_pairs": len(pairs),
    }
