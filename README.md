# Redrob AI — Intelligent Candidate Discovery & Ranking

A hybrid, multi-signal candidate ranking engine for the **Redrob India AI Challenge**.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- A Groq API key (free at https://console.groq.com)

### 1. Clone the project

```bash
git clone https://github.com/Vishal-V-D/The-Data-AI-Challenge.git
cd The-Data-AI-Challenge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `groq`, `streamlit`, `pandas`.

### 3. Add your API key

Create a `.env` file inside the repository root:

```bash
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. (One-time) Precompute LLM scores

This step calls the Groq API to evaluate the top-N candidates using an LLM.
It is **offline** — the result is saved locally and no API calls are made during final ranking.

```bash
python src/precompute_llm.py --top-n 1000 --output data/precomputed_llm_data.json
```

The script auto-resumes if interrupted. Groq free tier supports ~30 RPM.

### 5. Run the ranker (no network required)

```bash
python rank.py --output outputs/submission.csv
```

Runs in < 60 seconds on CPU-only, 16 GB RAM. Outputs a validated 100-candidate CSV.

### 6. Validate the submission

```bash
python "../extracted_data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" outputs/submission.csv
```

### 7. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8500 in your browser. The dashboard has 5 tabs:

| Tab | What it shows |
|---|---|
| 📊 Dashboard | Score distribution, dimension breakdown, location heatmap |
| 🏆 Top Candidates | Expandable cards for each ranked candidate with full score breakdown |
| 🔍 Explorer | Search and filter the scored candidate pool |
| 📁 Submission | Generate, validate, and download the submission CSV |
| 🧪 Evaluate Accuracy | Honeypot rate, score monotonicity, reasoning quality, dimension analysis |

---

## Dashboard Screenshots & Visuals

<img width="1918" height="910" alt="image" src="https://github.com/user-attachments/assets/27ee1b9f-5613-4f3f-bef5-be0477bff8d7" />
<img width="1917" height="912" alt="image" src="https://github.com/user-attachments/assets/fc5f0313-264c-45ae-8444-c233cfaaea24" />
<img width="1907" height="911" alt="image" src="https://github.com/user-attachments/assets/625b6214-d9e5-4eb1-a722-7385f9a68c7c" />
<img width="1890" height="908" alt="image" src="https://github.com/user-attachments/assets/4231c4e7-dea2-46c8-83a4-16e5a29781e0" />
<img width="1913" height="910" alt="image" src="https://github.com/user-attachments/assets/3722c25b-fd95-4f2e-87d1-4010377cf6a3" />
<img width="1552" height="822" alt="image" src="https://github.com/user-attachments/assets/d64eb2ce-ff2a-44ae-943f-3de4913286c8" />
<img width="1595" height="902" alt="image" src="https://github.com/user-attachments/assets/00dd031b-4aa2-457b-bdea-c2ef7a02bde9" />
<img width="1598" height="912" alt="image" src="https://github.com/user-attachments/assets/823c14af-d70d-415f-b5e1-c1ef23192143" />

https://drive.google.com/file/d/137dLWexPtliid1KYL24A2i601-Gb8tC9/view?usp=sharing
---


## Architecture

```
100,000 Candidates (candidates.jsonl)
         │
         ▼
┌─────────────────────────┐
│  Stage 1: Honeypot      │  66 impossible profiles filtered (7 rules,
│  Detector (honeypot.py) │  0 false positives vs sample_submission.csv)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 2: Rule-Based    │  Weighted scoring across 5 dimensions
│  Scorer (scorer.py)     │  + anti-pattern penalties
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐  (offline, run once — precompute_llm.py)
│  Stage 3: Groq LLM Eval │  llama-3.1-8b-instant evaluates top-1000
│  (precompute_llm.py)    │  → precomputed_llm_data.json
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 4: rank.py       │  Blends LLM score (35%) + rule score (65%)
│  Final Ranker           │  Outputs top-100 CSV in < 60s (CPU-only, no network)
└─────────────────────────┘
```

## Scoring Dimensions

| Dimension | Weight | Key Signals |
|---|---|---|
| Career Trajectory | 30% | Product vs consulting history; ML-specific titles; seniority 5-9 yrs |
| Technical Depth | 25% | Embeddings, vector DBs, hybrid search; skill duration weighting |
| Production Signals | 20% | GitHub activity score; Redrob assessment scores; profile completeness |
| Behavioral/Availability | 15% | Last active date; recruiter response rate; notice period; open-to-work |
| Location Bonus | 5% | Pune/Noida/NCR/Hyderabad/Mumbai/Bangalore; willing to relocate |

**Anti-pattern penalties** (up to −40%): pure consulting career, CV/robotics-only without NLP, inactivity > 6 months, keyword stuffing.

## Honeypot Detection (7 rules, 66 caught, 0 false positives)

| Rule | Condition | Catches |
|---|---|---|
| R1 | Job `duration_months` > actual date span + 3 months | Duration inflation |
| R2 | ≥3 `expert`-level skills with `duration_months == 0` | Fake expertise |
| R3 | Headline years-of-experience differs from profile by > 2 years | Free-text mismatch |
| R4 | Summary years-of-experience differs from profile by > 2 years | Free-text mismatch |
| R5 | Profile declares > 3 yrs experience but total work history < 1 year | Ghost history |
| R6 | Declared `years_of_experience` exceeds career span (from earliest job date) by > 3 years | Impossible timeline |
| R7 | Sum of all job `duration_months` exceeds declared `years_of_experience × 12` by > 48 months | Time-travel history |

**Validation**: All 7 rules verified against `sample_submission.csv` — **0 false positives**. The 66 flagged candidates have logically impossible profiles (e.g., 14.1 yrs declared experience but earliest job started 4.7 yrs ago).

## File Structure

```
.
├── rank.py                     # Main submission script (CPU-only, no network)
├── app.py                      # Streamlit dashboard (5 tabs)
├── requirements.txt            # groq, streamlit, pandas
├── README.md
├── .gitignore                  # Git ignore rules for keys and OS metadata
├── submission_metadata.yaml    # Participant metadata & methodology details
├── .env                        # GROQ_API_KEY=gsk_xxx (not committed)
├── src/
│   ├── __init__.py
│   ├── honeypot.py             # 7-rule honeypot detector (66 caught, 0 FP)
│   ├── scorer.py               # Multi-signal rule-based scorer & reasoning fallback
│   └── precompute_llm.py       # Offline Groq evaluator (run once)
├── data/
│   └── precomputed_llm_data.json   # Precomputed LLM scores (committed in repo)
└── outputs/
    └── submission.csv          # Final validated top-100 submission output
```

## Design Decisions

- **Why Groq instead of Gemini?** Groq's free tier provides ~30 RPM vs Gemini's 20 RPD, allowing candidates to be pre-evaluated offline rapidly in batches.
- **Why no live API calls in rank.py?** The challenge spec bans network access during the final ranking step. LLM scores are precomputed and cached locally.
- **Why 1000 candidates for LLM eval?** The coarse rule-based scorer has high recall. Evaluating the top 1000 ensures every candidate in the final top-100 shortlist has a rich, LLM-generated reasoning — not just a rule-based fallback.
- **Why 35% LLM weight?** LLM reasoning is powerful but can be noisy on sparse profiles. 35% gives meaningful signal without over-relying on a single source.
- **Consulting penalty**: Per-company check, not global. TCS + Flipkart gets partial credit.
- **Sparse profiles**: Missing fields default to neutral (0), not penalised.
