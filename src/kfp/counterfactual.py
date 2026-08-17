"""Counterfactual pair generation.

This is the instrument that answers the question the previous paper could not:
*does the model actually use the client profile, or has it memorised
`question type -> product`?*

For a scenario S with client C, we perturb exactly one attribute of C to get
C'. We keep the pair only if the rule engine's gold decision CHANGES. A model
that ignores the profile will give the same answer to both members of the
pair and score 0 on counterfactual correctness, no matter how high its
semantic similarity is.

Report two numbers:
  * counterfactual_correctness = share of pairs where the model's answer
    changes in the direction the rules require;
  * profile_sensitivity        = share of pairs where the answer changes at all.
The gap between them is diagnostic: high sensitivity with low correctness
means the model reacts to the profile but reacts wrongly.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from .schemas import ClientProfile, Scenario

# Each perturbation targets a specific eligibility rule so that the resulting
# pair has a documented causal reason for flipping.
PERTURBATIONS = {
    "income_down": lambda p, rng: replace(
        p, income_kzt=max(85000, int(p.income_kzt * 0.45))
    ),
    "income_up": lambda p, rng: replace(
        p, income_kzt=int(p.income_kzt * 2.2)
    ),
    "credit_score_down": lambda p, rng: replace(
        p, credit_score=max(300, p.credit_score - 120)
    ),
    "credit_score_up": lambda p, rng: replace(
        p, credit_score=min(850, p.credit_score + 120)
    ),
    "age_up": lambda p, rng: replace(p, age=min(80, p.age + 25)),
    "age_down": lambda p, rng: replace(p, age=max(18, p.age - 15)),
    "employment_unstable": lambda p, rng: replace(
        p, employment="unemployed", income_stability=0.2
    ),
    "children_added": lambda p, rng: replace(
        p, children=max(1, p.children + 1),
        family_status="married_with_kids" if p.family_status == "married_no_kids" else p.family_status,
    ),
    "children_removed": lambda p, rng: replace(
        p, children=0,
        family_status="married_no_kids" if p.family_status == "married_with_kids" else p.family_status,
    ),
    "debt_load_up": lambda p, rng: replace(
        p, existing_debt_service_kzt=int(p.income_kzt * 0.45)
    ),
}


def make_counterfactuals(
    profile: ClientProfile,
    feats: dict[str, float],
    scenarios: list[Scenario],
    catalog: dict,
    eligibility_module,
    suitability_module,
    seed: int,
    max_per_scenario: int = 2,
) -> tuple[list[ClientProfile], list[Scenario]]:
    rng = np.random.default_rng(seed)
    new_profiles: list[ClientProfile] = []
    new_scenarios: list[Scenario] = []

    names = list(PERTURBATIONS.keys())

    for sc in scenarios:
        kept = 0
        order = list(rng.permutation(names))
        for pname in order:
            if kept >= max_per_scenario:
                break
            cf_profile = PERTURBATIONS[pname](profile, rng)
            if cf_profile == profile:
                continue

            cf_id = f"{profile.client_id}-CF-{pname}"
            cf_profile = replace(cf_profile, client_id=cf_id)

            verdicts = eligibility_module.evaluate_all(cf_profile, catalog)
            eligible = [v.product_id for v in verdicts if v.eligible]
            gold_id, gold_action = suitability_module.gold_decision(
                catalog, eligible, cf_profile, feats, sc.intent
            )

            # keep only pairs where the decision genuinely flips
            if gold_id == sc.gold_product_id and gold_action == sc.gold_action:
                continue

            ineligible = [
                {"product_id": v.product_id, "violated_rules": v.violated_rules}
                for v in verdicts if not v.eligible
            ]

            new_profiles.append(cf_profile)
            new_scenarios.append(
                Scenario(
                    scenario_id=f"{sc.scenario_id}-CF-{pname}",
                    client_id=cf_id,
                    intent=sc.intent,
                    question_kk=sc.question_kk,
                    question_ru=sc.question_ru,
                    question_mixed=sc.question_mixed,
                    gold_product_id=gold_id,
                    gold_action=gold_action,
                    eligible_products=eligible,
                    ineligible_products=ineligible,
                    rationale_facts=[],   # filled by build script
                    is_negative_case=(gold_action == "refuse"),
                    is_cold_start=sc.is_cold_start,
                    counterfactual_of=sc.scenario_id,
                    perturbed_attribute=pname,
                )
            )
            kept += 1

    return new_profiles, new_scenarios
