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
python src/precompute_llm.py --top-n 200 --output data/precomputed_llm_data.json
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

Open http://localhost:8501 in your browser. The dashboard has 5 tabs:

| Tab | What it shows |
|---|---|
| 📊 Dashboard | Score distribution, dimension breakdown, location heatmap |
| 🏆 Top Candidates | Expandable cards for each ranked candidate with full score breakdown |
| 🔍 Explorer | Search and filter the scored candidate pool |
| 📁 Submission | Generate, validate, and download the submission CSV |
| 🧪 Evaluate Accuracy | Honeypot rate, score monotonicity, reasoning quality, dimension analysis |

---

## Dashboard Screenshots & Visuals


---


## Architecture

```
100,000 Candidates (candidates.jsonl)
         │
         ▼
┌─────────────────────────┐
│  Stage 1: Honeypot      │  53 impossible profiles filtered (0 false positives
│  Detector (honeypot.py) │  verified against sample_submission.csv)
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
│  Stage 3: Groq LLM Eval │  llama-3.3-70b evaluates top-1500 candidates
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

## Honeypot Detection (5 rules, 0 false positives)

| Rule | Condition |
|---|---|
| R1 | Job `duration_months` > actual date span + 3 months |
| R2 | ≥3 `expert`-level skills with `duration_months == 0` |
| R3 | Headline years-of-experience differs from profile by > 2 years |
| R4 | Summary years-of-experience differs from profile by > 2 years |
| R5 | Profile declares > 3 yrs experience but total work history < 1 year |

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
│   ├── honeypot.py             # 5-rule honeypot detector
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
- **Why 200 candidates for LLM eval?** The coarse rule-based scorer has high recall. Evaluating the top 200 is sufficient to obtain high-quality rankings and reasons for the final top 100 shortlist.
- **Why 35% LLM weight?** LLM reasoning is powerful but can be noisy on sparse profiles. 35% gives meaningful signal without over-relying on a single source.
- **Consulting penalty**: Per-company check, not global. TCS + Flipkart gets partial credit.
- **Sparse profiles**: Missing fields default to neutral (0), not penalised.
