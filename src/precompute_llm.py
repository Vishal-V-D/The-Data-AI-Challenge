"""
precompute_llm.py - Offline LLM evaluation using Groq API (free tier, ~30 RPM).

Run ONCE before submitting. Writes precomputed_llm_data.json.
Auto-resumes from existing output file if interrupted.

Usage:
    cd solution
    python src/precompute_llm.py --top-n 1500 --output data/precomputed_llm_data.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# ── Path setup (must be first) ────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent   # …/solution/src
_ROOT    = _SRC_DIR.parent                   # …/solution

sys.path.insert(0, str(_SRC_DIR))

from honeypot import is_honeypot
from scorer  import score_candidate

# ── Config ────────────────────────────────────────────────────────────────────

CANDIDATES_PATH = (
    _ROOT.parent
    / "extracted_data"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

JD_SUMMARY = """
Role: Senior AI Engineer — Founding Team at Redrob AI (Series A, Pune/Noida).
Needs 5-9 years of hands-on ML/AI PRODUCTION experience — specifically:
- Embeddings-based retrieval (sentence-transformers, BGE, E5, etc.) deployed to real users
- Vector DB / hybrid search (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch, OpenSearch)
- Strong Python; code quality matters
- Evaluation frameworks for ranking (NDCG, MRR, MAP, A/B testing)
- Product-company background preferred; pure consulting only = disqualifier
- "Shipper" mindset — ships working systems fast, not pure researcher
- Location: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR, Bangalore OK
- Notice period: sub-30 days preferred

Disqualifiers: pure research roles, recent LLM wrapper experience only,
no production code last 18 months, pure consulting career, CV/speech/robotics without NLP.
"""

PROMPT_TEMPLATE = """You are a senior technical recruiter evaluating a candidate for this role:

{jd}

Candidate profile:
ID: {cid}
Headline: {headline}
Years of experience: {yoe}
Current title: {title} at {company}
Location: {location}
Notice period: {notice} days

Career history (most recent first):
{history}

Key skills: {skills}

GitHub activity score: {github}/100  (-1 = not linked)
Redrob assessment scores: {assessments}
Last active: {last_active}
Recruiter response rate: {resp_rate}

Evaluate genuine fit. Consider:
1. REAL production experience with embeddings + vector search?
2. Shipper with product-company experience?
3. Strong behavioral signals (active, responsive, short notice)?
4. Red flags (pure consulting, CV-only, research-only, long inactivity)?

Return ONLY valid JSON (no markdown fences, no extra text):
{{
  "llm_score": <float 0.0 to 1.0>,
  "reasoning": "<1-2 factual sentences with specific details (company, skill, yrs, score); note any gaps>"
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_api_key(env_path: Path | None = None) -> str:
    """Load GROQ_API_KEY from .env file or environment variable."""
    paths = [env_path, _ROOT / ".env", _ROOT.parent / ".env"]
    for p in paths:
        if p and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k.strip() == "GROQ_API_KEY":
                        return v.strip()
    return os.environ.get("GROQ_API_KEY", "")


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Test that the Groq API key is valid by making a minimal call."""
    if not api_key or not api_key.startswith("gsk_"):
        return False, "GROQ_API_KEY not found or invalid format (must start with gsk_)."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            model=GROQ_MODELS[0],
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True, "API key valid."
    except Exception as e:
        return False, f"API key validation failed: {str(e)[:200]}"


def build_prompt(cand: Dict[str, Any]) -> str:
    p = cand.get("profile", {})
    s = cand.get("redrob_signals", {})
    history = cand.get("career_history", [])
    skills  = cand.get("skills", [])

    hist_lines = [
        f"  - {j.get('title')} at {j.get('company')} "
        f"({j.get('duration_months', 0)} months): "
        f"{(j.get('description') or '')[:200]}"
        for j in history[:5]
    ]
    skill_names = [f"{sk['name']}({sk.get('proficiency','?')})" for sk in skills[:12]]
    assessments = s.get("skill_assessment_scores") or {}
    assess_str  = ", ".join(f"{k}={v:.0f}" for k, v in assessments.items()) or "none"

    return PROMPT_TEMPLATE.format(
        jd=JD_SUMMARY,
        cid=cand["candidate_id"],
        headline=p.get("headline", ""),
        yoe=p.get("years_of_experience", "?"),
        title=p.get("current_title", ""),
        company=p.get("current_company", ""),
        location=p.get("location", ""),
        notice=s.get("notice_period_days", "?"),
        history="\n".join(hist_lines) or "  (none)",
        skills=", ".join(skill_names),
        github=s.get("github_activity_score", -1),
        assessments=assess_str,
        last_active=s.get("last_active_date", "?"),
        resp_rate=s.get("recruiter_response_rate", "?"),
    )


def call_groq(client, prompt: str, model: str, retries: int = 10) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                text  = "\n".join(lines).rstrip("`").strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 30 + attempt * 15
                print(f"    Rate-limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error (attempt {attempt+1}): {err[:150]}")
                time.sleep(5 * (attempt + 1))
    return {"llm_score": 0.5, "reasoning": "LLM evaluation unavailable for this candidate."}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute Groq LLM scores.")
    parser.add_argument("--top-n",     type=int, default=1500)
    parser.add_argument("--output",    type=str, default=str(_ROOT / "data" / "precomputed_llm_data.json"))
    parser.add_argument("--candidates",type=str, default=str(CANDIDATES_PATH))
    parser.add_argument("--model",     type=str, default=GROQ_MODELS[1])
    parser.add_argument("--sleep",     type=float, default=2.1,
                        help="Seconds between requests (free tier: ~30 RPM ≈ 2s)")
    args = parser.parse_args()

    # -- 1. API key validation --------------------------------------------------
    print("Validating Groq API key...")
    api_key = load_api_key()
    ok, msg = validate_api_key(api_key)
    if not ok:
        print(f"ERROR: {msg}")
        sys.exit(1)
    print(f"  [OK] {msg}")

    from groq import Groq
    client = Groq(api_key=api_key)

    # ── 2. Auto-resume ────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Found {len(existing)} existing entries - will skip already-evaluated candidates.")

    # ── 3. Coarse scoring to pick top-N ──────────────────────────────────
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        sys.exit(1)

    print(f"\nStage 1: Coarse scoring all candidates to pick top {args.top_n}...")
    coarse: list = []
    total = honeypot_count = 0

    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            cand = json.loads(line)
            flagged, _ = is_honeypot(cand)
            if flagged:
                honeypot_count += 1
                continue
            score, _ = score_candidate(cand)
            coarse.append((score, cand["candidate_id"], cand))

    coarse.sort(key=lambda x: (-x[0], x[1]))
    top_candidates = coarse[:args.top_n]
    print(f"  Scanned: {total}  |  Honeypots filtered: {honeypot_count}")
    print(f"  Top {len(top_candidates)} selected for LLM evaluation.")

    # -- 4. LLM evaluation -------------------------------------------------
    print(f"\nStage 2: LLM evaluation with model={args.model}...")
    results = dict(existing)
    to_eval = [(s, cid, c) for s, cid, c in top_candidates if cid not in results]
    print(f"  Need to evaluate: {len(to_eval)} candidates  (sleeping {args.sleep}s between calls)")

    for i, (_, cid, cand) in enumerate(to_eval):
        prompt = build_prompt(cand)
        print(f"  [{i+1}/{len(to_eval)}] {cid}...", end=" ", flush=True)
        result = call_groq(client, prompt, args.model)

        llm_score = max(0.0, min(1.0, float(result.get("llm_score", 0.5))))
        reasoning = str(result.get("reasoning", ""))[:500]
        results[cid] = {"llm_score": llm_score, "reasoning": reasoning}
        print(f"score={llm_score:.3f}")

        # Incremental save every 50
        if (i + 1) % 50 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [Checkpoint] Saved {len(results)} entries.")

        time.sleep(args.sleep)

    # -- 5. Final save -----------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {len(results)} LLM evaluations saved to {output_path}")


if __name__ == "__main__":
    main()
