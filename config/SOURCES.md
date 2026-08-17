# Source register

Every calibration parameter must appear here with its origin and retrieval
date. Reviewers check this; an unsourced number is an invented number.

Status: **RESOLVED** = official primary source recorded · **OPEN** = still a placeholder

---

## RESOLVED

### Income — Bureau of National Statistics RK
Q1 2026: mean monthly nominal wage **461,486 KZT**, median **331,527 KZT**.
https://stat.gov.kz/ru/news/zarabotnaya-plata-za-i-kvartal-2026-goda/ · accessed 2026-08-07

sigma is *derived*, not chosen: for a lognormal, mean/median = exp(sigma²/2),
so sigma = sqrt(2·ln(461486/331527)) = **0.81332**.
Validated: employed-segment median 328,000 (err 1.06%), mean 460,206 (err 0.28%).

⚠️ BNS wage statistics cover **employed workers only**. Self-employed, retired
and unemployed incomes are modelling assumptions via declared multipliers.
This must be stated in Limitations.

### Deposit products — Halyk Bank, ALCO minutes No. 75 of 12.06.2026
Rates effective 15.06.2026. This is a primary, dated, board-approved document —
exactly what a gold standard should rest on.
https://halykbank.kz/storage/app/media/Deposits/2026/6/Interest%20rates%20on%20deposits%20of%20individuals.pdf

| Product | Type | Term (KZT) | Rate KZT | AEIR |
|---|---|---|---|---|
| QORJIN | term, replenishable, no partial withdrawal | 3 / 6 / 12 | 17.10 / 15.37 / 14.50 | 18.51 / 16.50 / 15.50 |
| Maximum | savings, no prolongation, no replenish/withdraw | 3 / 6 / 12 / 24 | 16.67 / 15.80 / 14.93 / 12.15 | 18.00 / 17.00 / 16.00 / 12.85 |
| Universal | non-term, replenish + partial withdrawal | 6 / 12 | 14.32 / 14.06 | 15.30 / 15.00 |
| AQYL | educational (state programme) | 36 / 60 | 14.30 / 13.90 | 15.50 / 15.04 |

Minimum amounts (product pages, accessed 2026-08-07):
Universal 1,000 KZT / 10 USD / 10 EUR · Maximum 15,000 KZT / 50 USD · QORJIN 50,000 KZT / 50 USD

USD rates: QORJIN 0.7% (6m) / 0.98% (12m); Maximum 0.7 / 0.98 / 0.98; Universal 0.98%. EUR: Universal 0.98%.

**Bonus:** the PDF carries the full rate history back to 2019. That enables a
genuinely time-varying product catalogue — the model must track rates that
actually changed, not a frozen snapshot. No other Kazakh benchmark has this.

### Deposit guarantee — KDIF
20,000,000 KZT for KZT deposits, 10,000,000 KZT equivalent for FX.
⚠️ Figure came from an aggregator — **confirm on kdif.kz before citing.**

### Halyk product taxonomy (from official site navigation, accessed 2026-08-07)
- Deposits: Universal · Maximum · QORJIN · Halyk SHANYRAQ · AQYL
- Loans: no-collateral up to 8,000,000 ₸ · consumer loan · refinancing
- Installment: Halyk Installment Plan · Technodom / Sulpak / Mechta / Alser / Evrika / Viled
- Mortgage: Digital Mortgage · 7-20-25 · developer-partner housing · Halyk Sale
- Auto: digital auto loan · soft car loan
- Debit cards: Halyk Bonus Digital · Sinooil Digital · Halyk Easy · Special Allowance · Visa Signature · UnionPay Diamond · Mastercard World Elite · Visa Infinite Ultra · Mastercard World Elite Imperial
- Credit cards: My!Card · Black Premium · Visa Infinite
- Other: metal account · Halyk Invest · Halyk FX · gold · National Fund for Children

---

### Regions — Taldau, 2025 (RESOLVED)
- **SKZH 252101** — average monthly nominal wage per employee, by region, KZT, annual 2012–2025
- **SKZH 251401** — average listed number of employees, by region, persons, annual 2018–2025

`income_mult` = regional wage / national wage (443,315 KZT)
`weight` = regional headcount / national headcount (4,033,000)

Consistency check to report in the appendix:
- regional headcounts sum to **exactly** the national figure (100.000%)
- headcount-weighted wage = 443,498 vs published 443,315 (**+0.04%**)
- headcount-weighted multiplier = 1.0004, so the normalisation step is a no-op —
  an independent confirmation that the derivation is correct

Range: Mangystau 1.405 (highest) → Turkistan 0.720 (lowest).
⚠️ Break in series: South Kazakhstan oblast ends 2017; Turkistan + Shymkent from 2018;
Zhetysu, Ulytau, Abai from 2022. Regional time series is NOT continuous — state this.
⚠️ `urban` flag is still unsourced.

---

## OPEN — you must collect these

### Age structure — BNS demographic statistics
https://stat.gov.kz/ru/industries/social-statistics/demography/
Need: population by 5-year age group, 18+. Download the xlsx, do not retype.

### Spending structure — BNS household budget survey
Table: "Структура денежных расходов обследованных домашних хозяйств"
https://stat.gov.kz/ru/industries/labor-and-income/stat-life/dynamic-tables/
⚠️ Page is JS-rendered; links are not machine-readable. Download manually,
or use **taldau.stat.gov.kz** (the interactive data warehouse — better, it has
stable table IDs you can cite).
Known anchor: meat and meat products 18.3% and bread/cereals 7.6% of food
spending, Q3 2025, survey of 11.7k households.

### Employment structure — BNS
https://stat.gov.kz/ru/industries/labor-and-income/stat-empt-unempl/
Need: employed / self-employed / unemployed shares. Note that BNS "самозанятые"
does not map cleanly onto the categories used here — document the mapping.

### Minimum wage 2026
adilet.zan.kz (budget law) or enbek.gov.kz.

### Banking penetration — National Bank RK
https://nationalbank.kz — retail deposit and loan statistics.
Individual-level penetration may not be published; if only aggregate volumes
exist, say so and derive shares explicitly as an assumption.

### Credit score distribution
**No public Kazakh scorecard exists.** Do not pretend otherwise. Either state
Normal(640, 95) as a declared assumption and run a sensitivity analysis, or
replace the score rule with observable proxies (existing debt service, payment
history) that you can actually source.

### Credit product PRICING — halykbank.kz/credits (RESOLVED, 2026-08-07)

| Product | Rate | AEIR | Cap | Term |
|---|---|---|---|---|
| Кредит без залога | 17.5–38% | 28.8–45.3% | 8,000,000 ₸ | 5 y |
| Рефинансирование | 17.5–38% | 28.8–45.3% | 8,000,000 ₸ | 5 y |
| Кредит на потреб. цели | — | — | depends on income | 10 y |
| Цифровая ипотека | — | — | no cap | — |
| «7-20-25» | 7% | 7.24–7.80% | — | 25 y, 20% down |
| Строящееся жильё | 1.5–18.0% | 1.51–19.55% | — | 20 y, 20% down |
| Цифровой автокредит | — | — | 30,000,000 ₸ | — |
| Автокредит льготный (БРК) | 4% | 4.10–7.50% | 10,000,000 ₸ | 7 y |

### Eligibility rules for loans, cards, mortgage — STILL OPEN
The rate PDF covers deposits only. Age, income and employment requirements for
credit products are on the individual product pages and in the tariff documents
at https://halykbank.kz/documents — these need manual extraction.
This is the single largest remaining gap: credit products are where compliance
violations matter most.
