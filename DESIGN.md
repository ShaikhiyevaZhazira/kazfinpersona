# KazFinPersona — project contract

## Paper
**Working title:** Does Fine-Tuning Actually Personalize? Measuring Profile
Sensitivity and Constraint Compliance in Kazakh Financial LLM Assistants

**Framing:** personalization under constraints in a low-resource financial
domain. NOT "we applied an LLM to Kazakh banking".

## The one rule that must never be broken

**Gold labels come from `src/kfp/eligibility.py` + `src/kfp/suitability.py`,
never from a language model.**

The previous version of this research generated its dataset with
Qwen2.5-3B-Instruct, fine-tuned Qwen2.5-3B-Instruct on it, and scored the
result against Qwen2.5-3B-Instruct's own outputs. That is self-distillation,
not evaluation, and it is the likely reason the paper was rejected.

An LLM may only ever touch **surface realisation** of Kazakh text. If a change
would let a model's output influence a label, reject the change.

## Contributions

1. **KazFinPersona** — benchmark with rule-derived gold labels, negative
   cases, counterfactual pairs, cold-start clients, and kk/ru/code-switched
   question variants.
2. **Method** — behavioural segmentation → group-level LoRA adapters + a
   constraint-verification layer.
3. **Metrics** — profile sensitivity, counterfactual correctness, constraint
   violation rate, fact-level grounding, code-switching degradation. Plus the
   personalization ↔ compliance trade-off curve (headline figure).

## Experimental protocol (non-negotiable)

- Three seeds minimum. Report mean ± std, never a single point estimate.
- Paired significance tests (Wilcoxon or paired t) for every comparison.
- Bootstrap 95% CI on all headline numbers.
- Client-level splits. Counterfactuals are test-only.
- Every run writes `results/<exp_id>/{config.yaml, metrics.json, predictions.jsonl}`.

## Baselines to beat

zero-shot Qwen / GPT · persona-prompt · TALLRec-style instruction tuning ·
single-adapter LoRA (the previous paper's method) · **ours**

Include at least one Kazakh-centric model (KazLLM-8B or Sherkala-8B) or state
explicitly in Limitations why compute prevented it. Reviewers will ask.

## Stack

Python 3.11+, numpy, scipy, pandas, PyYAML, PyTorch, PEFT, transformers,
scikit-learn, pytest.

## Style

- Config in YAML/JSON, never hard-coded constants.
- Deterministic given a seed.
- Every new metric ships with a unit test on a hand-checked toy case.
