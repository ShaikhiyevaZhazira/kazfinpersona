"""Tests for parser v3 (lenient mode) and the v1.0 metric additions.
Place in tests/ next to the existing parser tests."""
import json, math
from kfp.eval.parsing import AnswerParser
from kfp.eval import metrics as M

CAT = {"products": [
    {"product_id": "hb_mortgage_digital", "name_kk": "Цифровая ипотека", "name_ru": "Цифровая ипотека"},
    {"product_id": "hb_deposit_universal", "name_kk": "Универсальный депозит", "name_ru": "Универсальный депозит"},
    {"product_id": "hb_deposit_aqyl", "name_kk": "AQYL", "name_ru": "AQYL депозит"},
    {"product_id": "hb_card_debit", "name_kk": "Halyk Bonus (дебеттік карта)", "name_ru": "Halyk Bonus (дебетовая карта)"},
    {"product_id": "hb_card_credit", "name_kk": "My!Card (кредитная карта)", "name_ru": "My!Card (кредитная карта)"},
]}

def test_strict_misses_typo_lenient_finds_it():
    t = "Подойдёт hb_mortage_digital."
    assert AnswerParser(CAT).parse(t)["unparseable"] is True
    assert AnswerParser(CAT, lenient=True).parse(t)["product_id"] == "hb_mortgage_digital"

def test_lenient_never_fuzzy_beyond_one_edit():
    assert AnswerParser(CAT, lenient=True).parse("hb_mortgg_digital")["unparseable"] is True

def test_lenient_ambiguous_id_rejected():
    # "hb_card_a3" is one edit from BOTH hb_card_a1 and hb_card_a2 -> not unique
    cat = {"products": [{"product_id": "hb_card_a1", "name_kk": "", "name_ru": ""},
                        {"product_id": "hb_card_a2", "name_kk": "", "name_ru": ""}]}
    assert AnswerParser(cat, lenient=True).parse("hb_card_a3")["unparseable"] is True

def test_lenient_short_name_before_parenthesis():
    t = "Рекомендую Halyk Bonus для оплаты."
    assert AnswerParser(CAT).parse(t)["unparseable"] is True
    assert AnswerParser(CAT, lenient=True).parse(t)["product_id"] == "hb_card_debit"

def test_strict_unchanged_on_exact_ids():
    assert AnswerParser(CAT).parse("HB Deposit Universal")["product_id"] == "hb_deposit_universal"

def test_f1_zero_when_recall_zero():
    recs = [dict(gold_action="refuse", pred_action="recommend"),
            dict(gold_action="recommend", pred_action="recommend")]
    r = M.refusal_metrics(recs)
    assert r["refusal_recall"] == 0 and r["refusal_f1"] == 0.0

def test_multi_product_and_single_pma():
    recs = [dict(gold_action="recommend", gold_product_id="a", pred_product_id="a",
                 recommended_ids=["a", "b"], unparseable=False),
            dict(gold_action="recommend", gold_product_id="a", pred_product_id="a",
                 recommended_ids=["a"], unparseable=False)]
    assert M.multi_product_rate(recs) == 0.5
    assert M.product_match_single(recs) == 0.5

def test_hard_rules_include_age_variants():
    assert {"age_max_by_gender", "age_max_at_maturity"} <= M.HARD_RULES

def test_wilson_bounds():
    lo, hi = M.wilson(0, 72)
    assert lo == 0.0 and 0.04 < hi < 0.06
