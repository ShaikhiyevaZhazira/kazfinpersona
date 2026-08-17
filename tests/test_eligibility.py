"""Hand-checked unit tests for the rule engine.

These are deliberately toy cases with answers worked out by hand. Every new
rule must arrive with a test like this — the rule engine defines the gold
labels, so a silent bug here corrupts the entire benchmark.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfp.eligibility import check_product, load_catalog
from kfp.schemas import ClientProfile

CATALOG = load_catalog(ROOT / "config" / "products_halyk.json")
BY_ID = {p["product_id"]: p for p in CATALOG["products"]}


def mk(**kw):
    base = dict(
        client_id="T", age=35, gender="f", region="Almaty", urban=True,
        employment="employed", income_kzt=500_000, income_stability=0.95,
        family_status="married_with_kids", children=2, credit_score=700,
        language_profile="kk_dominant", existing_products=["hb_card_debit"],
        existing_debt_service_kzt=0, tenure_months=24,
    )
    base.update(kw)
    return ClientProfile(**base)


def test_child_deposit_requires_children():
    p = BY_ID["hb_deposit_aqyl"]
    assert check_product(mk(children=1), p).eligible
    v = check_product(mk(children=0, family_status="single"), p)
    assert not v.eligible and "requires_children" in v.violated_rules


def test_mortgage_income_floor():
    p = BY_ID["hb_mortgage_digital"]
    assert check_product(mk(income_kzt=900_000), p).eligible
    v = check_product(mk(income_kzt=100_000), p)
    assert not v.eligible and "income_min" in v.violated_rules


def test_credit_card_age_bounds():
    p = BY_ID["hb_card_credit"]
    assert "age_min" in check_product(mk(age=19), p).violated_rules
    assert "age_max" in check_product(mk(age=70), p).violated_rules


def test_excludes_if_already_held():
    p = BY_ID["hb_card_debit"]
    v = check_product(mk(existing_products=["hb_card_debit"]), p)
    assert not v.eligible and "excludes_if_holds" in v.violated_rules
    assert check_product(mk(existing_products=[]), p).eligible


def test_brokerage_requires_debit_card():
    p = BY_ID["hb_invest"]
    v = check_product(mk(existing_products=[]), p)
    assert "requires_holds" in v.violated_rules


def test_dti_blocks_overleveraged_client():
    p = BY_ID["hb_loan_no_collateral"]
    v = check_product(mk(existing_debt_service_kzt=400_000), p)
    assert not v.eligible and "max_dti" in v.violated_rules


def test_verdict_always_explains_itself():
    for prod in CATALOG["products"]:
        v = check_product(mk(age=17, income_kzt=90_000, credit_score=310,
                             children=0, employment="unemployed"), prod)
        if not v.eligible:
            assert v.violated_rules and v.explanation_ru
