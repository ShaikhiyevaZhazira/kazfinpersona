"""Scenario construction.

Each scenario is a benchmark item: a client, an intent, a question in three
language conditions (kk / ru / code-switched), and a gold decision computed
by the rule engine.

The question templates below are seeds. Two things must happen before
publication:
  1. A native Kazakh speaker reviews and expands them (financial register,
     morphology, natural phrasing). Report inter-annotator agreement.
  2. An LLM may paraphrase them for surface diversity — but paraphrases are
     kept in a separate field and NEVER leak across the train/test boundary.
     Surface realisation is all an LLM is allowed to touch; the label is not
     negotiable.
"""
from __future__ import annotations

import numpy as np

from .schemas import ClientProfile, Scenario

# intent -> (kk template, ru template, code-switched template)
INTENT_TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "save": [
        ("Ақшамды қалай жинасам болады?",
         "Как мне лучше накопить деньги?",
         "Ақшамды қалай накопить етсем болады?"),
        ("Маған қандай депозит ашқан тиімді?",
         "Какой депозит мне выгоднее открыть?",
         "Маған қандай депозит открыть еткен тиімді?"),
        ("Айлық табысымның бір бөлігін сақтағым келеді, не ұсынасыз?",
         "Хочу откладывать часть зарплаты, что посоветуете?",
         "Зарплатамның бір бөлігін откладывать еткім келеді, не ұсынасыз?"),
    ],
    "education_savings": [
        ("Балама оқуға ақша жинағым келеді, қандай нұсқа бар?",
         "Хочу копить на образование ребёнка, какие есть варианты?",
         "Балама оқуға ақша жинағым келеді, қандай варианттар бар?"),
        ("Балаларға арналған жинақ бағдарламасы бар ма?",
         "Есть ли программа накоплений для детей?",
         "Балаларға арналған накопительная программа бар ма?"),
    ],
    "finance_purchase": [
        ("Тұрмыстық техника алғым келеді, қалай төлеген дұрыс?",
         "Хочу купить бытовую технику, как лучше оплатить?",
         "Бытовая техника алғым келеді, қалай оплатить еткен дұрыс?"),
        ("Ірі сатып алуға қаражат керек, не ұсынасыз?",
         "Нужны средства на крупную покупку, что посоветуете?",
         "Крупная покупкаға қаражат керек, не ұсынасыз?"),
    ],
    "liquidity_buffer": [
        ("Айдың соңында ақша жетпей қалады, не істеуге болады?",
         "В конце месяца не хватает денег, что можно сделать?",
         "Айдың соңында деньги жетпей қалады, не істеуге болады?"),
    ],
    "grow_capital": [
        ("Жинақтарымды өсіргім келеді, қандай нұсқалар бар?",
         "Хочу приумножить накопления, какие есть варианты?",
         "Накоплениеларымды өсіргім келеді, қандай варианттар бар?"),
    ],
    "buy_home": [
        ("Пәтер алғым келеді, қандай шарттар бар?",
         "Хочу купить квартиру, какие есть условия?",
         "Квартира алғым келеді, қандай условиялар бар?"),
    ],
    "finance_vehicle": [
        ("Көлік алуға несие бар ма?",
         "Есть ли кредит на покупку автомобиля?",
         "Машина алуға кредит бар ма?"),
    ],
    "hedge_fx": [
        ("Долларда сақтағым келеді, не ұсынасыз?",
         "Хочу хранить в долларах, что посоветуете?",
         "Долларда хранить еткім келеді, не ұсынасыз?"),
    ],
    "retirement": [
        ("Зейнетақыма қосымша жинақ жасай алам ба?",
         "Могу ли я делать дополнительные пенсионные накопления?",
         "Пенсияма қосымша накопление жасай алам ба?"),
    ],
    "refinance_debt": [
        ("Қолданыстағы несиемді қайта қаржыландыруға бола ма?",
         "Можно ли рефинансировать мой текущий кредит?",
         "Қолданыстағы кредитімді рефинансировать етуге бола ма?"),
        ("Ай сайынғы төлемім ауыр, не істеуге болады?",
         "Ежемесячный платёж тяжёлый, что можно сделать?",
         "Ай сайынғы платежім ауыр, не істеуге болады?"),
    ],
    "daily_payments": [
        ("Күнделікті төлемдерге қандай карта ыңғайлы?",
         "Какая карта удобна для повседневных платежей?",
         "Күнделікті платеждерге қандай карта ыңғайлы?"),
    ],
}

# which intents are plausible for a client, given their situation
def plausible_intents(profile: ClientProfile, feats: dict[str, float]) -> list[str]:
    out = ["save", "daily_payments"]
    if profile.children > 0:
        out.append("education_savings")
    if feats.get("free_cashflow", 0) > 60000:
        out += ["grow_capital", "retirement"]
    if feats.get("fx_share", 0) > 0.01:
        out.append("hedge_fx")
    if feats.get("retail_share", 0) > 0.08:
        out.append("finance_purchase")
    if feats.get("savings_rate", 0) < 0.05:
        out.append("liquidity_buffer")
    if profile.age >= 23 and feats.get("transport_share", 0) > 0.07:
        out.append("finance_vehicle")
    if profile.income_kzt > 350000 and not any(
        p.startswith("hb_mortgage") for p in profile.existing_products
    ):
        out.append("buy_home")
    if feats.get("debt_service_ratio", 0) > 0.12:
        out.append("refinance_debt")
    return sorted(set(out))


def rationale_facts(
    catalog: dict,
    gold_id: str | None,
    gold_action: str,
    verdicts: list,
    profile: ClientProfile,
) -> list[str]:
    """Atomic facts a correct answer must be consistent with.

    Used by the grounding metric instead of fuzzy string overlap: an answer is
    checked fact-by-fact, which is far more defensible than 'does the product
    name appear somewhere in the text'.
    """
    by_id = {p["product_id"]: p for p in catalog["products"]}
    facts: list[str] = []
    if gold_action == "refuse":
        blocked = [v for v in verdicts if not v.eligible]
        if blocked:
            facts.append(f"no_eligible_product_for_intent")
            facts.append(f"primary_blocking_rule={blocked[0].violated_rules[0]}")
        return facts
    if gold_id:
        p = by_id[gold_id]
        facts.append(f"product_id={gold_id}")
        attrs = p.get("attributes", {})
        if "rate_annual_pct" in attrs:
            facts.append(f"rate_annual_pct={attrs['rate_annual_pct']}")
        if "min_amount_kzt" in attrs:
            facts.append(f"min_amount_kzt={attrs['min_amount_kzt']}")
        if "term_months" in attrs:
            facts.append(f"term_months={attrs['term_months']}")
        if gold_action == "refer_to_advisor":
            facts.append("ambiguous_requires_advisor")
    return facts


def build_scenarios(
    profile: ClientProfile,
    feats: dict[str, float],
    catalog: dict,
    eligibility_module,
    suitability_module,
    seed: int,
    max_per_client: int = 3,
    cold_start: bool = False,
) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    verdicts = eligibility_module.evaluate_all(profile, catalog)
    eligible = [v.product_id for v in verdicts if v.eligible]
    ineligible = [
        {"product_id": v.product_id, "violated_rules": v.violated_rules}
        for v in verdicts if not v.eligible
    ]

    intents = plausible_intents(profile, feats)
    rng.shuffle(intents)
    intents = intents[:max_per_client]

    out: list[Scenario] = []
    for k, intent in enumerate(intents):
        templates = INTENT_TEMPLATES.get(intent)
        if not templates:
            continue
        q_kk, q_ru, q_mix = templates[int(rng.integers(len(templates)))]
        gold_id, gold_action = suitability_module.gold_decision(
            catalog, eligible, profile, feats, intent
        )
        out.append(
            Scenario(
                scenario_id=f"{profile.client_id}-S{k}",
                client_id=profile.client_id,
                intent=intent,
                question_kk=q_kk,
                question_ru=q_ru,
                question_mixed=q_mix,
                gold_product_id=gold_id,
                gold_action=gold_action,
                eligible_products=eligible,
                ineligible_products=ineligible,
                rationale_facts=rationale_facts(
                    catalog, gold_id, gold_action, verdicts, profile
                ),
                is_negative_case=(gold_action == "refuse"),
                is_cold_start=cold_start,
            )
        )
    return out
