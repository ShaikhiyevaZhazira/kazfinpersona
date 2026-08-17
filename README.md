# KazFinPersona

A benchmark and generation pipeline for **personalization under constraints**
in Kazakh-language retail banking.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_all.sh      # full pipeline, no GPU, a few minutes
python tests/run_tests.py    # 18 tests, no pytest needed
```

**The oracle must score 1.000 on all 15 metrics** on action accuracy, product match and
counterfactual correctness, and 0.000 on constraint violations. It consults the
rule engine directly, so anything else means the pipeline is broken — fix that
before running a real model, because a plumbing bug looks exactly like a model
failure in the results table.

### Individual steps

```bash
python scripts/build_dataset.py --n-clients 2000 --seed 42 --out data/v0.5
python scripts/fit_segments.py  --data data/v0.5 --seed 42
python scripts/run_baseline.py  --data data/v0.5 --backend oracle
python scripts/evaluate.py      --data data/v0.5 --simulate
```

With a GPU:

```bash
python scripts/run_baseline.py --data data/v0.5 --backend hf \
    --model Qwen/Qwen2.5-7B-Instruct --context profile --language kk
```

Outputs `clients.jsonl`, `clients_counterfactual.jsonl`, `transactions.jsonl`,
`life_events.jsonl`, `scenarios.jsonl`, plus `split_report.json`,
`validation.json` and `validation_table.tex` for the paper appendix.

---

## Why this design

### Gold labels are computed, not generated

`eligibility.py` applies the published product terms as machine-checkable
rules and returns, for every product, either *eligible* or a list of violated
rule ids. `suitability.py` then ranks the eligible, on-goal products with a
transparent additive score.

No language model is involved in producing a label. This is the single most
important property of the benchmark: it makes product-match accuracy and
constraint-violation rate mean something, and it eliminates the
self-distillation loop of the earlier pipeline.

### Three gold actions, not one

`recommend` · `refuse` · `refer_to_advisor`

A dataset in which every client qualifies for the product being discussed
cannot measure compliance at all. Roughly 43% of scenarios here are refusals,
where the correct behaviour is to decline and name the blocking rule.
`refer_to_advisor` fires when the top two candidates are within a score
margin — forcing a single label there would inject noise.

### Counterfactual pairs

For each test scenario the generator perturbs exactly one client attribute
(income, credit score, age, employment, dependants, debt load) and keeps the
pair **only if the rule engine's decision changes**.

A model that ignores the client profile answers both members identically and
scores zero, no matter how fluent it is. This is the instrument the previous
paper lacked: it had no way to distinguish real personalization from a
memorised `question type → product` mapping.

### Three language conditions

Every question exists in Kazakh, Russian, and a code-switched variant. The
gap between them is the code-switching degradation metric — a genuinely
under-studied phenomenon, and directly relevant since deployed Kazakh
assistants operate bilingually.

### Time and drift

Transactions span an 18-month window with dated entries and injected life
events (job loss, new child, relocation, income shock). Behavioural features
use 90-day half-life exponential decay. Without a time axis, preference drift
— an open problem named in both RecGPT and INVESTMATE — cannot be studied.

### Validation, not assertion

`validate.py` runs KS and chi-square goodness-of-fit tests of the generated
population against the declared reference distributions and emits a booktabs
table. Note the honest claim: this proves the generator samples from the model
it *declares*. That the model matches Kazakhstan rests on the official
statistics cited in `config/kz_stats.yaml`.

---

## Before the final build — required

1. **Resolve every `[VERIFY]` in `config/kz_stats.yaml`** against the Bureau
   of National Statistics and the National Bank of RK. Record source URL and
   retrieval date. Placeholders make the validation table meaningless.
2. **Replace every product term in `config/products_halyk.json`** with the
   officially published Halyk terms. Record the product page URL and date.
3. **Native-speaker review of `scenarios.py` templates.** Financial register,
   morphology, natural phrasing. Report inter-annotator agreement.
4. **Check product coverage.** In the current draft some products never become
   gold (consumer loan, auto loan, brokerage, voluntary pension) because their
   thresholds are strict or a competing product always outranks them. Once the
   real terms are in, re-check the gold distribution and widen intent coverage
   if a product is unreachable.

## Human annotation protocol

Sample 400 scenarios stratified by gold action and intent. Three annotators —
at least one banking practitioner. They judge: is the gold product the right
recommendation, and is the refusal reason correct? Report Fleiss' κ and the
agreement rate with the rule engine. **Disagreement is a finding, not a
failure**: it quantifies how far a rule-based gold standard sits from expert
judgement, and that number belongs in the paper.

## Known open items

1. **14 of 16 products have `assumed` eligibility rules.** Pricing is sourced;
   borrower requirements are not. Until they are collected from the product
   pages and tariff documents, the compliance metric measures *the assumptions
   in this repo*, not Halyk policy. The build script prints a warning; the
   paper must say so in Limitations.
2. **Age, spending and employment structure are still placeholders** in
   `kz_stats.yaml`. Income and regions are sourced and validated.
3. **Question templates need a native speaker.** Currently 1–3 per intent,
   written by a non-native. Target 8–10, with inter-annotator agreement.
4. **Counterfactual perturbations are unbalanced** — `employment_unstable`
   dominates. Balance before drawing conclusions about profile sensitivity,
   or the finding is about one axis only.
5. **Three products are never gold** (`hb_invest`, `hb_loan_consumer`,
   `hb_mortgage_developer`) — always outranked by a competitor on the same
   goal. Re-check once real thresholds are in.

## Roadmap

- [x] Profile / transaction / life-event generation
- [x] Deterministic eligibility engine
- [x] Suitability ranking with hard goal gate
- [x] Counterfactual pair generation
- [x] Leak-safe splitting with assertions
- [x] Distribution validation + LaTeX export
- [ ] LLM surface realisation of answers (Kazakh) — labels stay frozen
- [x] Behavioural clustering → segment assignment (`scripts/fit_segments.py`)
- [x] Metrics package (`src/kfp/eval/`)
- [x] Metric validation via synthetic policies (`scripts/evaluate.py --simulate`)
- [x] Prompt/backend/runner framework (`prompts.py`, `backends.py`, `run_baseline.py`)
- [ ] Baselines on a real model (needs GPU)
- [ ] Group-LoRA + constraint-verification layer
- [ ] Three-seed experiment runner with significance testing
