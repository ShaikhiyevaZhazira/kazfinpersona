"""Turning a free-form model answer into a structured prediction.

Every metric downstream depends on this module, so it is deliberately
conservative and fully auditable:

  * it never guesses a product from a partial match,
  * it distinguishes "recommended product X" from "mentioned product X while
    explaining why it is unavailable",
  * it returns an `unparseable` flag rather than silently defaulting, because
    a model that produces unusable output should be penalised visibly and not
    hidden inside an accuracy denominator.

The previous pipeline detected hallucination by checking whether a product
name appeared anywhere in the text. That conflates "recommends a deposit" with
"explains that a deposit is not available to you", and it cannot see a
fabricated interest rate attached to a real product name.
"""
from __future__ import annotations

import re
import unicodedata

# Phrases that indicate the surrounding product mention is a *rejection*,
# not a recommendation. Extend after native-speaker review.
# Markers must be SPECIFIC. An early version listed the bare Kazakh word
# "төмен" (low), which matches inside the entirely ordinary phrase
# "ең төменгі сома" (minimum amount). Every correct recommendation that quoted
# a minimum amount was therefore reclassified as a refusal: the oracle scored
# 0.355 product-match instead of 1.000. The failure was uniform across systems,
# so it would not have shown up as an anomaly in a comparison table -- it would
# just have made every result quietly wrong.
#
# Rule: only multi-word phrases or morphologically unambiguous negatives.
# Anything that could appear inside a neutral banking phrase is banned.
NEGATION_MARKERS_KK = [
    "жарамайсыз", "сәйкес келмейді", "қол жетімсіз", "қолжетімсіз",
    "мүмкін емес", "талаптарға сай емес", "ұсына алмаймын",
    "рәсімдей алмайсыз", "табысыңыз жеткіліксіз", "ұсынбаймын",
    "бере алмаймыз", "мақұлданбайды",
]
NEGATION_MARKERS_RU = [
    "не подходит", "недоступн", "не можем предложить", "не соответству",
    "отказ", "не хватает", "ниже минимального", "не проходите",
    "не одобр", "не можем оформить",
]

REFUSAL_MARKERS = NEGATION_MARKERS_KK + NEGATION_MARKERS_RU
ADVISOR_MARKERS = [
    "маман", "кеңесші", "бөлімшеге", "консультант", "специалист",
    "отделение", "менеджер",
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


class AnswerParser:
    def __init__(self, catalog: dict):
        self.catalog = catalog
        self.aliases: dict[str, list[str]] = {}
        for p in catalog["products"]:
            names = [p.get("name_kk", ""), p.get("name_ru", "")]
            names = [_norm(n) for n in names if n]
            self.aliases[p["product_id"]] = names
        self.by_id = {p["product_id"]: p for p in catalog["products"]}

    # ------------------------------------------------------------ products
    def find_mentions(self, text: str) -> list[tuple[str, int]]:
        """Return (product_id, char_offset) for every product name found."""
        t = _norm(text)
        found: list[tuple[str, int]] = []
        for pid, names in self.aliases.items():
            for n in names:
                idx = t.find(n)
                if idx >= 0:
                    found.append((pid, idx))
                    break
        return sorted(found, key=lambda kv: kv[1])

    def _is_negated(self, text: str, offset: int, window: int = 120) -> bool:
        t = _norm(text)
        lo = max(0, offset - window)
        hi = min(len(t), offset + window)
        ctx = t[lo:hi]
        return any(m in ctx for m in REFUSAL_MARKERS)

    # -------------------------------------------------------------- action
    def parse(self, text: str) -> dict:
        if not text or not text.strip():
            return {"action": None, "product_id": None, "unparseable": True,
                    "mentions": [], "asserted_numbers": []}

        t = _norm(text)
        mentions = self.find_mentions(text)
        recommended = [(pid, off) for pid, off in mentions
                       if not self._is_negated(text, off)]

        advisor = any(m in t for m in ADVISOR_MARKERS)
        refusal_language = any(m in t for m in REFUSAL_MARKERS)

        if recommended:
            action = "refer_to_advisor" if advisor else "recommend"
            product_id = recommended[0][0]
        elif refusal_language or mentions:
            action = "refuse"
            product_id = None
        elif advisor:
            action = "refer_to_advisor"
            product_id = None
        else:
            return {"action": None, "product_id": None, "unparseable": True,
                    "mentions": [pid for pid, _ in mentions],
                    "asserted_numbers": self.extract_numbers(text)}

        return {
            "action": action,
            "product_id": product_id,
            "unparseable": False,
            "mentions": [pid for pid, _ in mentions],
            "recommended_ids": [pid for pid, _ in recommended],
            "asserted_numbers": self.extract_numbers(text),
        }

    # ------------------------------------------------------------- numbers
    @staticmethod
    def extract_numbers(text: str) -> list[float]:
        """Numeric claims the model makes (rates, amounts, terms).

        Checked against the catalog by the grounding metric. A fabricated
        interest rate attached to a correctly-named product is a hallucination
        that name-matching cannot see.
        """
        out: list[float] = []
        for m in re.finditer(r"(\d[\d\s.,]*)\s*(%|пайыз|процент|теңге|тг|ай|мес)?", text):
            raw = m.group(1).replace(" ", "").replace(",", ".").rstrip(".")
            if not raw:
                continue
            try:
                out.append(float(raw))
            except ValueError:
                continue
        return out
