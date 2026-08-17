"""Deterministic eligibility engine.

This module is the scientific core of the benchmark. Gold labels are produced
HERE, by transparent rules over the published product terms — never by a
language model. That removes the self-distillation circularity that made the
earlier `semantic similarity` numbers uninterpretable.

Every rejection carries a machine-readable rule id, which gives you:
  * ground truth for product-match accuracy,
  * the denominator for the constraint-violation-rate metric,
  * an auditable explanation string for the qualitative analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import ClientProfile, EligibilityVerdict

_RULE_TEXT = {
    "age_min": "клиент моложе минимального возраста для продукта",
    "age_max": "клиент старше максимального возраста для продукта",
    "income_min": "подтверждённый доход ниже минимального порога",
    "employment": "тип занятости не допускается для этого продукта",
    "credit_score_min": "кредитный скоринг ниже требуемого",
    "max_dti": "долговая нагрузка превышает допустимый предел",
    "requires_children": "продукт требует наличия детей",
    "excludes_if_holds": "продукт уже оформлен у клиента",
    "requires_holds": "требуется наличие сопутствующего продукта",
}


def load_catalog(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _estimate_new_payment(product: dict, profile: ClientProfile) -> float:
    """Rough monthly payment used only for the DTI rule.

    Deliberately simple and documented in the paper: DTI checks compare
    existing obligations plus a standardised notional payment for the
    candidate product against income.
    """
    cat = product["category"]
    attrs = product.get("attributes", {})
    if cat in ("deposit", "investment", "pension"):
        return 0.0
    if cat == "card":
        limit = min(attrs.get("limit_max_kzt", 0), profile.income_kzt * 3)
        return limit * 0.05
    if cat == "installment":
        return min(attrs.get("amount_max_kzt", 0), profile.income_kzt * 2) / 12.0
    if cat == "loan":
        amount = min(attrs.get("amount_max_kzt", profile.income_kzt * 12),
                     profile.income_kzt * 12)
        term = max(attrs.get("term_months", [36])[0], 12)
        rate = attrs.get("rate_annual_pct", 20.0) / 100.0 / 12.0
        if rate <= 0:
            return amount / term
        return amount * rate / (1 - (1 + rate) ** (-term))
    return 0.0


def check_product(profile: ClientProfile, product: dict) -> EligibilityVerdict:
    rules = product["eligibility"]
    violated: list[str] = []

    if rules.get("age_min") is not None and profile.age < rules["age_min"]:
        violated.append("age_min")
    if rules.get("age_max") is not None and profile.age > rules["age_max"]:
        violated.append("age_max")
    if profile.income_kzt < rules.get("income_min_kzt", 0):
        violated.append("income_min")
    allowed = rules.get("employment_allowed")
    if allowed and profile.employment not in allowed:
        violated.append("employment")
    if rules.get("credit_score_min") is not None and profile.credit_score < rules["credit_score_min"]:
        violated.append("credit_score_min")
    if rules.get("requires_children") and profile.children < 1:
        violated.append("requires_children")
    for pid in rules.get("excludes_if_holds", []):
        if pid in profile.existing_products:
            violated.append("excludes_if_holds")
            break
    req = rules.get("requires_holds", [])
    if req and not any(p in profile.existing_products for p in req):
        violated.append("requires_holds")
    if rules.get("max_dti") is not None:
        new_payment = _estimate_new_payment(product, profile)
        dti = (profile.existing_debt_service_kzt + new_payment) / max(profile.income_kzt, 1)
        if dti > rules["max_dti"]:
            violated.append("max_dti")

    explanation = ""
    if violated:
        explanation = "; ".join(_RULE_TEXT.get(r, r) for r in violated)

    return EligibilityVerdict(
        product_id=product["product_id"],
        eligible=not violated,
        violated_rules=violated,
        explanation_ru=explanation,
    )


def evaluate_all(profile: ClientProfile, catalog: dict) -> list[EligibilityVerdict]:
    return [check_product(profile, p) for p in catalog["products"]]


def eligible_ids(profile: ClientProfile, catalog: dict) -> list[str]:
    return [v.product_id for v in evaluate_all(profile, catalog) if v.eligible]
