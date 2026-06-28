# Redrob Hackathon — Intelligent Candidate Ranker

## What this does

Ranks 100,000 candidates for the **Senior AI Engineer (Founding Team)** role at Redrob AI using a weighted multi-component scorer with behavioral signal modifiers. Runs in **< 5 minutes on CPU** with no network calls and no GPU.

## Architecture

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Honeypot Filter      (impossible profiles → skip)  │
│  Disqualifier Check   (consulting-only, non-tech)   │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Component Scorers                                  │
│    ① Title & Career Fit     (28%)                   │
│    ② Core Skills Match      (25%)                   │
│    ③ Experience Band        (15%)                   │
│    ④ Behavioral Signals     (15%)                   │
│    ⑤ Location & Availability(12%)                   │
│    ⑥ Education              (5%)                    │
└─────────────────────────────────────────────────────┘
      │
      ▼
  Weighted sum → Sort → Top 100 → submission.csv
```

### Component details

| Component | Weight | Key logic |
|---|---|---|
| Title & Career | 28% | Rewards ML/AI titles; production deployment tokens in descriptions; penalizes pure consulting |
| Skills | 25% | Trust-weighted: `proficiency × log(1+endorsements) × log(1+duration_months)` matched against required skill list |
| Experience | 15% | 5-9 yr band gets 1.0; quality-adjusted by fraction of ML-relevant months |
| Behavioral | 15% | Recency, response rate, open-to-work, notice period, GitHub activity |
| Location | 12% | India + Pune/Noida/Delhi = 1.0; relocation willing = 0.85 |
| Education | 5% | Tier 1-2 bonus; CS/ML field bonus; not decisive |

### Why NOT keyword stuffing

Skills scoring uses a **trust multiplier**: `proficiency × log(endorsements) × log(duration_months)`. A keyword-stuffer with 10 "expert" skills, 0 endorsements, 0 months = near-zero weight. A candidate with 3 "advanced" skills, 50 endorsements, 36 months = high weight.

### Honeypot detection

Three signals: (1) ≥5 expert skills all with 0 duration + 0 endorsements, (2) total career months >1.8× stated years of experience, (3) ≥8 expert skills on a <40% complete profile. Detected candidates get score = -0.5 and never appear in top 100.

### Disqualifiers (heavy penalty, not elimination)

- Entire career at TCS / Infosys / Wipro / Accenture / Cognizant (etc.)
- Current title is Marketing, Operations, HR, Sales, etc.
- CV/vision-only background with no NLP/IR experience

These get score multiplied by 0.25 so they fall to ranks 90-100 only if top-100 has < 100 qualified candidates.

## Setup

```bash
python -m pip install -r requirements.txt
```

No models to download. No embeddings to pre-compute. Pure Python + stdlib.

## Reproduce submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- **Runtime:** ~40 seconds for 100K candidates on a single CPU core
- **Memory:** ~2 GB peak
- **Network:** None
- **GPU:** Not used

## Validate

```bash
python validate_submission.py submission.csv
```

## Files

```
rank.py                       # Main ranker — the only file that matters
validate_submission.py        # Copied from hackathon bundle
requirements.txt              # No external deps beyond stdlib
submission_metadata.yaml      # Metadata template filled in
README.md                     # This file
```
