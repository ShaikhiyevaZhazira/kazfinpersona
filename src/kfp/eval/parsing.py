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

Two modes
---------
strict (default, the primary result)
    exact match of a product id (also with spaces instead of underscores) or a
    display name; brackets and quotation marks are ignored.

lenient (sensitivity analysis only)
    additionally accepts (a) a token that looks like a product id and is within
    one character edit of exactly ONE catalogue id (e.g. the systematic typo
    "hb_mortage_digital" for "hb_mortgage_digital"), and (b) the part of a
    display name before its parenthesis ("Halyk Bonus" for "Halyk Bonus
    (дебетовая карта)") when that part is unique in the catalogue.
    Report strict and lenient side by side: the gap is the share of answers
    lost to naming variants rather than to missing content.
"""
from __future__ import annotations

import re
import unicodedata

# Bump when the parsing rules change, so results can state which parser
# produced them. Scoring re-parses the saved answer text, so a parser change
# never requires regenerating model outputs.
#   1: initial
#   2: brackets/quotes stripped before matching; product_id also matched with
#      underscores replaced by spaces ("HB Deposit Universal")
#   3: optional lenient mode (one-edit product ids, unique short names)
PARSER_VERSION = 3

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
#
# NOTE: markers are deliberately NOT extended after looking at a weak model's
# degenerate Kazakh output. Doing so would tune the parser on test outputs.
# Any extension must come from a documented audit of unparseable answers.
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

# Brackets and quotation marks are removed on BOTH sides of the comparison
# (aliases and answer text), so "Halyk Bonus (дебеттік карта)" matches whether
# or not the model kept the parentheses.
_STRIP_CHARS = re.compile(r"[()«»\"“”„]")
_ID_TOKEN = re.compile(r"hb_[a-z0-9_]+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _STRIP_CHARS.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _within_one_edit(a: str, b: str) -> bool:
    """True if the edit distance between a and b is at most 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1:] == b[i + 1:]       # one substitution
    return a[i:] == b[i + 1:]               # one insertion in the longer string


class AnswerParser:
    def __init__(self, catalog: dict, lenient: bool = False):
        self.catalog = catalog
        self.lenient = lenient
        # The product_id itself MUST be an alias. Models that follow the
        # instruction to name catalogue products often emit the canonical id
        # ("hb_deposit_universal") rather than the display name. An earlier
        # version searched only name_kk/name_ru, so a fluent, correct,
        # instruction-following answer was scored `unparseable`.
        #
        # This penalised precisely the better-behaved models: KazLLM-8B was
        # reported at 66.7% unparseable while its answers were grammatical and
        # on-topic, purely because it referred to products by id.
        #
        # The id is also matched with underscores replaced by spaces, because
        # models write "HB Deposit Universal" as well as "hb_deposit_universal".
        # This is an exact match after a separator change, not a fuzzy match.
        self.aliases: dict[str, list[str]] = {}
        for p in catalog["products"]:
            pid = p["product_id"]
            names = [pid, pid.replace("_", " "),
                     p.get("name_kk", ""), p.get("name_ru", "")]
            names = [_norm(n) for n in names if n]
            self.aliases[pid] = sorted(set(names), key=len, reverse=True)
        self.by_id = {p["product_id"]: p for p in catalog["products"]}
        self.ids = [p["product_id"] for p in catalog["products"]]

        if lenient:
            self._add_short_names(catalog)

    def _add_short_names(self, catalog: dict) -> None:
        """Display name before its parenthesis, kept only if unique."""
        for p in catalog["products"]:
            pid = p["product_id"]
            for nm in (p.get("name_kk", ""), p.get("name_ru", "")):
                if "(" not in nm:
                    continue
                base = _norm(nm.split("(")[0])
                if len(base) < 8 or base in self.aliases[pid]:
                    continue
                others = [a for q, al in self.aliases.items() if q != pid for a in al]
                if any(base in a for a in others):
                    continue                  # ambiguous: skip
                self.aliases[pid].append(base)
            self.aliases[pid] = sorted(set(self.aliases[pid]), key=len, reverse=True)

    # ------------------------------------------------------------ products
    def _near_id(self, token: str) -> str | None:
        """The single catalogue id within one edit of `token`, else None."""
        hits = [pid for pid in self.ids if _within_one_edit(token, pid)]
        return hits[0] if len(hits) == 1 else None

    def find_mentions(self, text: str) -> list[tuple[str, int]]:
        """Return (product_id, char_offset) for every product name found."""
        t = _norm(text)
        found: list[tuple[str, int]] = []
        for pid, names in self.aliases.items():
            for n in names:
                if len(n) < 4:
                    continue          # too short to be an unambiguous mention
                idx = t.find(n)
                if idx >= 0:
                    found.append((pid, idx))
                    break
        if self.lenient:
            have = {pid for pid, _ in found}
            for m in _ID_TOKEN.finditer(t):
                pid = self._near_id(m.group(0).strip("_"))
                if pid and pid not in have:
                    found.append((pid, m.start()))
                    have.add(pid)
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
                    "mentions": [], "recommended_ids": [], "asserted_numbers": []}

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
                    "recommended_ids": [],
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
