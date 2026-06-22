"""
rank.py - Main submission ranker for Redrob Intelligent Candidate Discovery Challenge.

Usage (no network required during execution):
    cd solution
    python rank.py [--output outputs/submission.csv] [--llm-data data/precomputed_llm_data.json]

Produces: outputs/submission.csv with columns [candidate_id, rank, score, reasoning]
Runtime: < 60s on CPU-only, 16 GB RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent

from src.honeypot import is_honeypot
from src.scorer import score_candidate, generate_rule_reasoning

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATES = (
    _ROOT.parent
    / "extracted_data"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)
DEFAULT_LLM_DATA = _ROOT / "data" / "precomputed_llm_data.json"
DEFAULT_OUTPUT = _ROOT / "outputs" / "submission.csv"


# ---------------------------------------------------------------------------
# Main ranking pipeline
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates_path: Path,
    llm_data_path: Optional[Path],
    output_path: Path,
    top_n: int = 100,
) -> None:
    t0 = time.time()

    # Load precomputed LLM data
    llm_data: Dict[str, Dict] = {}
    if llm_data_path and llm_data_path.exists():
        with open(llm_data_path, encoding="utf-8") as f:
            llm_data = json.load(f)
        print(f"Loaded {len(llm_data)} LLM evaluations from {llm_data_path.name}")
    else:
        print("No LLM data found — using rule-based scoring only.")

    # Stream and score all candidates
    print(f"\nScoring candidates from {candidates_path.name}...")
    scored: List[Tuple[float, str, Dict, Dict, Optional[str]]] = []
    total = honeypots = skipped = 0

    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Stage 1: Honeypot filter
            flagged, _ = is_honeypot(cand)
            if flagged:
                honeypots += 1
                continue

            # Stage 2: Score
            cid = cand["candidate_id"]
            llm_entry = llm_data.get(cid)
            llm_score = llm_entry["llm_score"] if llm_entry else None

            final_score, breakdown = score_candidate(cand, llm_score)

            # Reasoning: prefer LLM, fall back to rule-based if LLM failed or is placeholder
            llm_reasoning = llm_entry.get("reasoning", "").strip() if llm_entry else ""
            if llm_reasoning and "unavailable" not in llm_reasoning.lower():
                reasoning = llm_reasoning
            else:
                reasoning = generate_rule_reasoning(cand, breakdown)

            scored.append((final_score, cid, cand, breakdown, reasoning))

    print(f"  Total: {total}, Honeypots filtered: {honeypots}, Parse errors: {skipped}")
    print(f"  Valid candidates scored: {len(scored)}")

    # Sort: descending score, tie-break by candidate_id ascending (per spec)
    # candidate_id format is CAND_XXXXXXX — lexicographic sort is numerically correct
    scored.sort(key=lambda x: (-round(x[0], 4), x[1]))
    top100 = scored[:top_n]

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, cand, breakdown, reasoning) in enumerate(top100, start=1):
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    elapsed = time.time() - t0
    print(f"\nOutput written to: {output_path}")
    print(f"Top candidate: {top100[0][1]} with score {top100[0][0]:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob AI challenge.")
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--llm-data", type=Path, default=DEFAULT_LLM_DATA,
        help="Path to precomputed_llm_data.json (optional)"
    )
    parser.add_argument(
        "--output", "--out", type=Path, default=DEFAULT_OUTPUT,
        help="Output CSV path"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of candidates to output (default: 100)"
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}")
        sys.exit(1)

    rank_candidates(args.candidates, args.llm_data, args.output, args.top_n)


if __name__ == "__main__":
    main()
