"""Hand-checked tests for the metric suite.

Each case has an answer worked out by hand. The suite defines the paper's
claims, so a silent bug here is worse than no metric at all.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kfp.eval import metrics as M
from kfp.eval.parsing import AnswerParser
from kfp.eligibility import load_catalog

CATALOG = load_catalog(ROOT / "config" / "products_halyk.json")


def rec(**kw):
    base = dict(scenario_id="s", client_id="c", intent="save",
                gold_action="recommend", gold_product_id="hb_deposit_qorjin",
                pred_action="recommend", pred_product_id="hb_deposit_qorjin",
                eligible_products=["hb_deposit_qorjin"], ineligible_products=[],
                rationale_facts=[], counterfactual_of=None, asserted_numbers=[],
                answer_text="")
    base.update(kw); return base


def test_pma_ignores_refusal_items():
    recs = [rec(), rec(gold_action="refuse", gold_product_id=None,
                       pred_action="refuse", pred_product_id=None)]
    assert M.product_match_accuracy(recs) == 1.0
    assert M.action_accuracy(recs) == 1.0


def test_cvr_flags_ineligible_recommendation():
    r = rec(pred_product_id="hb_mortgage_digital", eligible_products=["hb_deposit_qorjin"])
    assert M.constraint_violation_rate([r]) == 1.0
    assert M.constraint_violation_rate([rec()]) == 0.0


def test_hard_violation_separates_from_soft():
    soft = rec(pred_product_id="hb_loan_no_collateral", eligible_products=[],
               ineligible_products=[{"product_id": "hb_loan_no_collateral",
                                     "violated_rules": ["max_dti"]}])
    hard = rec(pred_product_id="hb_mortgage_digital", eligible_products=[],
               ineligible_products=[{"product_id": "hb_mortgage_digital",
                                     "violated_rules": ["age_min"]}])
    assert M.hard_violation_rate([soft]) == 0.0
    assert M.hard_violation_rate([hard]) == 1.0


def test_profile_blind_scores_zero_on_counterfactuals():
    base = rec(gold_product_id="hb_deposit_qorjin", pred_product_id="hb_deposit_qorjin")
    cf = rec(counterfactual_of="s", gold_action="refuse", gold_product_id=None,
             pred_action="recommend", pred_product_id="hb_deposit_qorjin")
    pairs = [(base, cf)]
    assert M.profile_sensitivity(pairs) == 0.0
    assert M.counterfactual_correctness(pairs) == 0.0


def test_sensitive_but_wrong_is_detected():
    """Changes the answer, but to the wrong product: PS=1, CC=0."""
    base = rec(pred_product_id="hb_deposit_qorjin")
    cf = rec(counterfactual_of="s", gold_product_id="hb_deposit_universal",
             pred_product_id="hb_installment")
    pairs = [(base, cf)]
    assert M.profile_sensitivity(pairs) == 1.0
    assert M.counterfactual_correctness(pairs) == 0.0
    assert M.sensitivity_correctness_gap(pairs) == 1.0


def test_refusal_recall():
    recs = [rec(gold_action="refuse", pred_action="refuse"),
            rec(gold_action="refuse", pred_action="recommend")]
    assert M.refusal_metrics(recs)["refusal_recall"] == 0.5


def test_parser_distinguishes_recommend_from_reject():
    p = AnswerParser(CATALOG)
    ok = p.parse("Сізге QORJIN ұсынамын, пайыздық мөлшерлеме 14.5%")
    assert ok["action"] == "recommend" and ok["product_id"] == "hb_deposit_qorjin"
    no = p.parse("Ипотека сізге жарамайсыз, табысыңыз жеткіліксіз")
    assert no["action"] == "refuse" and no["product_id"] is None


def test_parser_flags_empty_output():
    p = AnswerParser(CATALOG)
    assert p.parse("")["unparseable"] is True


def test_fact_fabrication_catches_invented_rate():
    good = rec(pred_product_id="hb_deposit_qorjin", asserted_numbers=[14.5])
    bad = rec(pred_product_id="hb_deposit_qorjin", asserted_numbers=[99.0])
    assert M.fact_grounding([good], CATALOG)["fact_fabrication_rate"] == 0.0
    assert M.fact_grounding([bad], CATALOG)["fact_fabrication_rate"] == 1.0


def test_parser_does_not_treat_minimum_amount_as_refusal():
    """Regression: the Kazakh word for 'low' matches inside 'minimum amount'.

    A bare 'төмен' marker turned every recommendation that quoted a minimum
    amount into a refusal, dropping the oracle from 1.000 to 0.355 on product
    match. Uniform across systems, therefore invisible in a comparison table.
    """
    p = AnswerParser(CATALOG)
    r = p.parse("Сізге QORJIN ұсынамын, жылдық мөлшерлеме 14.5%, "
                "ең төменгі сома 50000 теңге.")
    assert r["action"] == "recommend"
    assert r["product_id"] == "hb_deposit_qorjin"


def test_parser_still_detects_a_real_refusal():
    p = AnswerParser(CATALOG)
    r = p.parse("Ипотека ұсына алмаймын, талаптарға сай емессіз.")
    assert r["action"] == "refuse" and r["product_id"] is None


# --------------------------------------------------------------------------
# Regressions found by running the ORACLE through the metric suite. The oracle
# consults the rule engine directly, so any score below 1.000 is a metric bug
# by definition. Both of these were exactly that.
# --------------------------------------------------------------------------

def test_coverage_excludes_refusals_from_denominator():
    """A refusal names no product, so it cannot 'cover' a product fact.

    Counting refusals in the denominator scored 281 correct refusals as 0%
    and pulled the oracle down to 0.528.
    """
    rec_ok = rec(rationale_facts=["product_id=hb_deposit_qorjin"],
                 pred_product_id="hb_deposit_qorjin")
    refusal = rec(gold_action="refuse", gold_product_id=None,
                  pred_action="refuse", pred_product_id=None,
                  rationale_facts=["no_eligible_product_for_intent",
                                   "primary_blocking_rule=max_dti"],
                  answer_text="Ұсына алмаймын, себебі max_dti.")
    out = M.fact_grounding([rec_ok, refusal], CATALOG)
    assert out["fact_coverage"] == 1.0
    assert out["refusal_reason_coverage"] == 1.0


def test_product_name_digits_are_not_hallucinations():
    """'7-20-25' contains the numbers 7, 20 and 25.

    Seven is the rate and twenty the down payment; 25 is the term in YEARS
    while the catalogue stores 300 MONTHS. The oracle was flagged at 3.2%
    fabrication for quoting its own product's name.
    """
    r = rec(pred_product_id="hb_mortgage_7_20_25",
            asserted_numbers=[7.0, 20.0, 25.0],
            rationale_facts=["product_id=hb_mortgage_7_20_25"])
    out = M.fact_grounding([r], CATALOG)["fact_fabrication_rate"]
    # After removing the name digits there is nothing left to check, so the
    # metric is undefined for this single item rather than zero. Both outcomes
    # mean "no fabrication detected"; what must NOT happen is a positive rate.
    assert out != out or out == 0.0, out

    # and with a genuine claim alongside the name digits, still clean
    r2 = rec(pred_product_id="hb_mortgage_7_20_25",
             asserted_numbers=[7.0, 20.0, 25.0, 7.24],
             rationale_facts=["product_id=hb_mortgage_7_20_25"])
    assert M.fact_grounding([r2], CATALOG)["fact_fabrication_rate"] == 0.0


def test_unit_conversion_is_accepted():
    """300 months spoken as 25 years is true, not invented."""
    allowed = M._allowed_numbers(
        {"attributes": {"term_months": [300], "min_amount_kzt": 50000}})
    assert 25.0 in allowed      # 300 months -> years
    assert 50.0 in allowed      # tenge -> thousands


def test_real_fabrication_is_still_caught():
    r = rec(pred_product_id="hb_deposit_qorjin", asserted_numbers=[99.0],
            rationale_facts=["product_id=hb_deposit_qorjin"])
    assert M.fact_grounding([r], CATALOG)["fact_fabrication_rate"] == 1.0

def test_parser_ignores_brackets_in_product_names():
    p = AnswerParser(CATALOG)
    r = p.parse("Мен сізге Halyk Bonus (дебеттік карта) ұсынамын")
    assert r["product_id"] == "hb_card_debit"
