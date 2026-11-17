"""Prompt construction.

Every baseline differs in exactly one declared dimension: how much client
context the model sees, and whether the product catalogue is in the prompt.
Keeping that in a single module means the ablation is a *controlled* comparison
rather than four separately hand-tuned prompt engineering efforts -- which is
the usual reason ablation tables in LLM papers are not interpretable.

Context levels (this is the ablation axis, put it in the paper as Table 2):

  none      question only. Floor: anything above this is using the profile.
  persona   short natural-language profile sketch, the 2023-style approach.
  profile   structured profile fields.
  full      structured profile + behavioural features + segment id.

Catalogue modes:

  closed    no catalogue in the prompt; the model must rely on parametric
            knowledge. Reveals hallucinated product names.
  listed    product names and terms in the prompt.
  filtered  only products the client is actually eligible for.

`filtered` is not a baseline, it is an ORACLE upper bound: it hands the model
the output of the rule engine. Report it as such and never as a system.
"""
from __future__ import annotations

import json

SYSTEM_KK = (
    "Сіз Halyk Bank клиенттеріне арналған қаржылық көмекшісіз. "
    "Клиенттің жағдайына сай банк өнімін ұсыныңыз. "
    "Егер клиент өнім талаптарына сай келмесе, ұсынбаңыз және себебін түсіндіріңіз. "
    "Тек банк каталогындағы нақты өнімдерді атаңыз. Қысқа жауап беріңіз."
)

SYSTEM_RU = (
    "Вы финансовый ассистент для клиентов Halyk Bank. "
    "Предложите банковский продукт, подходящий ситуации клиента. "
    "Если клиент не соответствует требованиям продукта, не предлагайте его и объясните причину. "
    "Называйте только реальные продукты из каталога банка. Отвечайте кратко."
)

_EMPLOYMENT_RU = {
    "employed": "наёмный работник",
    "self_employed": "самозанятый",
    "business_owner": "владелец бизнеса",
    "student": "студент",
    "retired": "пенсионер",
    "unemployed": "без работы",
}

_FAMILY_RU = {
    "single": "не в браке",
    "married_no_kids": "в браке, без детей",
    "married_with_kids": "в браке, есть дети",
    "single_parent": "воспитывает детей один",
}


def persona_sketch(profile: dict) -> str:
    """The 2023-style approach: profile rendered as prose.

    Included precisely because it is what the previous paper did. If the full
    method does not beat this, that is a finding worth reporting honestly.
    """
    parts = [
        f"Клиент, {profile['age']} лет",
        _EMPLOYMENT_RU.get(profile["employment"], profile["employment"]),
        _FAMILY_RU.get(profile["family_status"], profile["family_status"]),
    ]
    if profile.get("children"):
        parts.append(f"детей: {profile['children']}")
    parts.append(f"регион: {profile['region']}")
    parts.append(f"доход около {profile['income_kzt']:,} тг в месяц".replace(",", " "))
    return ". ".join(parts) + "."


def structured_profile(profile: dict) -> str:
    fields = {
        "возраст": profile["age"],
        "занятость": _EMPLOYMENT_RU.get(profile["employment"], profile["employment"]),
        "доход_мес_тг": profile["income_kzt"],
        "регион": profile["region"],
        "семья": _FAMILY_RU.get(profile["family_status"], profile["family_status"]),
        "детей": profile["children"],
        "кредитный_скоринг": profile["credit_score"],
        "текущие_платежи_по_долгам_тг": profile["existing_debt_service_kzt"],
        "имеющиеся_продукты": profile["existing_products"] or ["нет"],
        "стаж_обслуживания_мес": profile["tenure_months"],
    }
    return json.dumps(fields, ensure_ascii=False, indent=None)


def behaviour_block(feats: dict, segment: int | None) -> str:
    b = {
        "свободный_денежный_поток_мес_тг": round(feats.get("free_cashflow", 0.0)),
        "норма_сбережений": round(feats.get("savings_rate", 0.0), 3),
        "волатильность_расходов": round(feats.get("spend_volatility", 0.0), 3),
        "доля_карточных_платежей": round(feats.get("card_share", 0.0), 3),
        "доля_валютных_операций": round(feats.get("fx_share", 0.0), 3),
        "доля_расходов_на_образование": round(feats.get("education_share", 0.0), 3),
        "доля_расходов_на_транспорт": round(feats.get("transport_share", 0.0), 3),
        "доля_обслуживания_долга": round(feats.get("debt_service_ratio", 0.0), 3),
    }
    if segment is not None:
        b["поведенческий_сегмент"] = segment
    return json.dumps(b, ensure_ascii=False)


import random

def catalogue_block(catalog: dict, allowed: list[str] | None = None,
                    shuffle_key: str | None = None) -> str:
    products = list(catalog["products"])
    if shuffle_key is not None:
        random.Random(shuffle_key).shuffle(products)
    lines = []
    for p in products:
        if allowed is not None and p["product_id"] not in allowed:
            continue
        a = p.get("attributes", {})
        bits = [f"{p['product_id']} — {p.get('name_ru', '')}"]
        if "rate_annual_pct" in a:
            bits.append(f"ставка {a['rate_annual_pct']}%")
        if "term_months" in a:
            bits.append(f"срок {a['term_months']} мес")
        if "min_amount_kzt" in a:
            bits.append(f"мин. сумма {a['min_amount_kzt']} тг")
        lines.append("; ".join(bits))
    return "\n".join(lines)


def build_prompt(
    scenario: dict,
    profile: dict,
    catalog: dict,
    feats: dict | None = None,
    segment: int | None = None,
    context: str = "profile",
    catalogue: str = "listed",
    language: str = "kk",
) -> tuple[str, str]:
    """Return (system, user). Deterministic given the inputs."""
    question = {
        "kk": scenario["question_kk"],
        "ru": scenario["question_ru"],
        "mixed": scenario["question_mixed"],
    }[language]

    blocks: list[str] = []

    if context == "persona":
        blocks.append("Клиент: " + persona_sketch(profile))
    elif context == "profile":
        blocks.append("Профиль клиента: " + structured_profile(profile))
    elif context == "full":
        blocks.append("Профиль клиента: " + structured_profile(profile))
        if feats is not None:
            blocks.append("Поведение по транзакциям: " + behaviour_block(feats, segment))
    elif context != "none":
        raise ValueError(f"unknown context level: {context}")
    
    order_key = scenario.get("counterfactual_of") or scenario["scenario_id"]

    if catalogue == "listed":
        blocks.append("Каталог продуктов:\n" + catalogue_block(catalog, shuffle_key=order_key))
    elif catalogue == "filtered":
        blocks.append(
            "Продукты, доступные этому клиенту:\n"
            + catalogue_block(catalog, scenario["eligible_products"], shuffle_key=order_key)
        )
        

    elif catalogue != "closed":
        raise ValueError(f"unknown catalogue mode: {catalogue}")

    blocks.append("Вопрос клиента: " + question)
    system = SYSTEM_KK if language in ("kk", "mixed") else SYSTEM_RU
    return system, "\n\n".join(blocks)
