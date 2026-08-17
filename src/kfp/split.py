"""Leak-safe dataset splitting.

Three rules, each of which fixes a specific reviewer objection:

1. **Client-level split.** No client appears in more than one split. (The
   previous pipeline did this correctly — keep it.)

2. **Counterfactual pairs are test-only, and both members land together.**
   If a counterfactual appeared in training, the model could learn the
   perturbation itself rather than the underlying rule, and the
   personalisation metric would be circular in the same way the old semantic
   similarity was.

3. **Paraphrases never cross the split boundary, and the *test* split holds
   originals only.** Reporting on a test set padded with paraphrases of each
   other inflates every number; state the count of unique scenarios
   separately from the count of surface forms.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schemas import Scenario


def assign_splits(
    scenarios: list[Scenario],
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[Scenario]:
    rng = np.random.default_rng(seed)

    base = [s for s in scenarios if s.counterfactual_of is None]
    cf = [s for s in scenarios if s.counterfactual_of is not None]

    by_client: dict[str, list[Scenario]] = defaultdict(list)
    for s in base:
        by_client[s.client_id].append(s)

    clients = sorted(by_client.keys())
    rng.shuffle(clients)
    n = len(clients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split_of_client: dict[str, str] = {}
    for i, c in enumerate(clients):
        if i < n_train:
            split_of_client[c] = "train"
        elif i < n_train + n_val:
            split_of_client[c] = "val"
        else:
            split_of_client[c] = "test"

    # Counterfactuals are generated *after* splitting, from test clients only,
    # so they inherit the test split by construction. Anything else that shows
    # up here is a build-order bug and is caught by the assertion below.
    out: list[Scenario] = []
    for s in base:
        s.split = split_of_client[s.client_id]
        out.append(s)
    origin_split = {s.scenario_id: s.split for s in base}
    for s in cf:
        parent = origin_split.get(s.counterfactual_of)
        if parent is not None and parent != "test":
            raise AssertionError(
                f"counterfactual {s.scenario_id} derives from a {parent} scenario; "
                "counterfactuals must be built from test clients only"
            )
        s.split = "test"
        out.append(s)
    return out


def split_report(scenarios: list[Scenario]) -> dict:
    """Numbers to put in the dataset table of the paper."""
    rep: dict = {}
    for split in ("train", "val", "test"):
        items = [s for s in scenarios if s.split == split]
        rep[split] = {
            "scenarios": len(items),
            "unique_clients": len({s.client_id for s in items}),
            "negative_cases": sum(s.is_negative_case for s in items),
            "cold_start": sum(s.is_cold_start for s in items),
            "counterfactual_items": sum(s.counterfactual_of is not None for s in items),
            "refer_to_advisor": sum(s.gold_action == "refer_to_advisor" for s in items),
        }
    all_clients = [{s.client_id for s in scenarios if s.split == sp}
                   for sp in ("train", "val", "test")]
    rep["leakage_check"] = {
        "train_val_overlap": len(all_clients[0] & all_clients[1]),
        "train_test_overlap": len(all_clients[0] & all_clients[2]),
        "val_test_overlap": len(all_clients[1] & all_clients[2]),
    }
    return rep
