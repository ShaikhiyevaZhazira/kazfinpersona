"""Dataclass schemas for KazFinPersona.

Everything the generator produces is a plain dataclass so that it can be
serialised to JSONL without surprises and diffed across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ClientProfile:
    client_id: str
    age: int
    gender: str
    region: str
    urban: bool
    employment: str
    income_kzt: int
    income_stability: float          # 0..1, drives volatility of salary inflows
    family_status: str
    children: int
    credit_score: int
    language_profile: str            # kk_dominant | ru_dominant | mixed
    existing_products: list[str]
    existing_debt_service_kzt: int   # monthly obligations already in place
    tenure_months: int               # relationship length with the bank
    segment: str | None = None       # filled later by behavioural clustering

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transaction:
    client_id: str
    date: str                        # ISO-8601 date
    category: str
    amount_kzt: int                  # positive = inflow, negative = outflow
    channel: str                     # card | transfer | cash | qr | autopay
    merchant_type: str | None = None
    currency: str = "KZT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LifeEvent:
    client_id: str
    date: str
    event: str
    magnitude: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EligibilityVerdict:
    """Deterministic output of the rule engine — the ground truth."""
    product_id: str
    eligible: bool
    violated_rules: list[str] = field(default_factory=list)
    explanation_ru: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    """One benchmark item.

    `gold_product_id` and `eligible_products` come from the rule engine,
    NOT from a language model. This is what breaks the self-distillation
    circularity of the earlier pipeline.
    """
    scenario_id: str
    client_id: str
    intent: str                      # save | finance_purchase | education_savings | ...
    question_kk: str
    question_ru: str
    question_mixed: str
    gold_product_id: str | None      # None => correct answer is a refusal
    gold_action: str                 # recommend | refuse | refer_to_advisor
    eligible_products: list[str]
    ineligible_products: list[dict[str, Any]]
    rationale_facts: list[str]       # atomic facts a correct answer must contain
    is_negative_case: bool
    is_cold_start: bool
    split: str = "unassigned"
    counterfactual_of: str | None = None
    perturbed_attribute: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
