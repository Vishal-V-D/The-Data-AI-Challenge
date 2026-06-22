"""
scorer.py - Multi-signal rule-based candidate scorer.

Scoring dimensions (total sums to 1.0 before LLM weight injection):
  D1  Career Trajectory  30%
  D2  Technical Depth    25%
  D3  Production Signals 20%
  D4  Behavioral/Avail   15%
  D5  Location Bonus      5%
  D6  Anti-pattern penalty (negative, applied post-sum)

The returned score is normalised to [0, 1].
"""

from __future__ import annotations

import re
from datetime import datetime, date
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants – tuned from JD + dataset analysis
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "wipro", "infosys", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl", "tech mahindra",
    "l&t infotech", "mphasis", "hexaware", "mindtree", "niit technologies",
}

PRODUCT_KEYWORDS = {
    "startup", "series a", "series b", "saas", "product company",
    "ai-native", "funded", "venture", "vc-backed",
}

ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "search engineer", "nlp engineer", "data scientist",
    "applied scientist", "ranking engineer", "retrieval engineer",
    "research engineer", "recommendation engineer", "llm engineer",
}

CORE_SKILLS = {
    "python", "embeddings", "vector database", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "sentence-transformers", "sentence transformers", "hugging face",
    "transformers", "retrieval", "ranking", "reranking", "re-ranking",
    "hybrid search", "bm25", "ndcg", "mrr", "map", "rag",
    "large language models", "llm", "fine-tuning", "lora", "qlora",
    "peft", "xgboost", "learning to rank", "neural ranking",
}

BONUS_SKILLS = {
    "pytorch", "tensorflow", "bert", "gpt", "openai",
    "langchain", "llamaindex", "chroma", "redis", "kafka",
    "spark", "ray", "triton", "vllm", "tgi",
}

CV_ONLY_SKILLS = {
    "opencv", "yolo", "object detection", "image segmentation",
    "computer vision", "facial recognition", "speech recognition",
    "asr", "tts", "robotics", "slam", "ros",
}

PREFERRED_LOCATIONS = {
    "noida", "pune", "delhi", "gurgaon", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "ncr",
}

_CURRENT_DATE = datetime(2026, 6, 22)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _contains_any(text: str, keywords: set) -> bool:
    t = _norm(text)
    return any(kw in t for kw in keywords)


def _count_matches(text: str, keywords: set) -> int:
    t = _norm(text)
    return sum(1 for kw in keywords if kw in t)


def _parse_date(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _days_since(date_str: str) -> int:
    d = _parse_date(date_str)
    if not d:
        return 9999
    return (_CURRENT_DATE - d).days


# ---------------------------------------------------------------------------
# Dimension scorers (each returns float 0-1)
# ---------------------------------------------------------------------------

def _score_career_trajectory(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    years_exp = profile.get("years_of_experience", 0) or 0
    score = 0.0

    # Seniority sweet-spot: 5-9 years
    if 5 <= years_exp <= 9:
        score += 0.35
    elif 4 <= years_exp < 5 or 9 < years_exp <= 12:
        score += 0.20
    elif 3 <= years_exp < 4 or 12 < years_exp <= 15:
        score += 0.10
    # <3 or >15 gets 0

    companies = [j.get("company", "") for j in history]
    titles_all = " ".join(j.get("title", "") for j in history)
    is_cons = [_contains_any(c, CONSULTING_FIRMS) for c in companies]

    # Product vs consulting
    if not any(is_cons):
        score += 0.30  # pure product background
    elif all(is_cons):
        score += 0.00  # pure consulting - no bonus (will be penalised in D6)
    else:
        score += 0.15  # hybrid – product + consulting

    # ML/AI titles in history
    ml_title_hits = _count_matches(titles_all, ML_TITLES)
    current_title = _norm(profile.get("current_title", ""))
    if any(t in current_title for t in ML_TITLES):
        ml_title_hits += 1
    score += min(ml_title_hits * 0.10, 0.35)

    return min(score, 1.0)


def _score_technical_depth(candidate: Dict[str, Any]) -> float:
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    
    # Build full text blob from skills + job descriptions
    skill_text = " ".join(_norm(s.get("name", "")) for s in skills)
    hist_text = " ".join(
        _norm(j.get("title", "") + " " + j.get("description", ""))
        for j in history
    )
    full_text = skill_text + " " + hist_text

    # Core skill hits weighted by duration
    core_hits = 0
    for s in skills:
        name_lower = _norm(s.get("name", ""))
        if any(kw in name_lower for kw in CORE_SKILLS):
            dur = s.get("duration_months", 0) or 0
            # Duration bonus: >30 months of a core skill is significant
            core_hits += 1
            if dur >= 30:
                core_hits += 1
            elif dur >= 12:
                core_hits += 0.5

    # Also count mentions in job descriptions (lower weight)
    core_hits += _count_matches(hist_text, CORE_SKILLS) * 0.3

    score = min(core_hits * 0.05, 0.80)

    # Bonus skill overlap
    bonus_hits = _count_matches(full_text, BONUS_SKILLS)
    score += min(bonus_hits * 0.02, 0.20)

    return min(score, 1.0)


def _score_production_signals(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # GitHub activity (0-100; -1 = not linked)
    github = signals.get("github_activity_score", -1)
    if github > 0:
        score += min(github / 100.0, 1.0) * 0.45
    # else 0 – no GitHub is a mild negative but not catastrophic

    # Redrob skill assessments
    assessments = signals.get("skill_assessment_scores", {}) or {}
    if assessments:
        best_score = max(assessments.values())
        score += min(best_score / 100.0, 1.0) * 0.30
        # Multiple assessments bonus
        score += min(len(assessments) * 0.05, 0.15)

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0) or 0
    score += (completeness / 100.0) * 0.10

    return min(score, 1.0)


def _score_behavioral(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # Recency of activity
    days_inactive = _days_since(signals.get("last_active_date", ""))
    if days_inactive <= 30:
        score += 0.30
    elif days_inactive <= 90:
        score += 0.20
    elif days_inactive <= 180:
        score += 0.10
    # > 6 months: 0

    # Recruiter responsiveness
    resp_rate = signals.get("recruiter_response_rate", 0) or 0
    score += resp_rate * 0.25

    # Notice period
    notice = signals.get("notice_period_days", 90) or 90
    if notice <= 30:
        score += 0.25
    elif notice <= 60:
        score += 0.15
    elif notice <= 90:
        score += 0.05

    # Open to work
    if signals.get("open_to_work_flag"):
        score += 0.10

    # Interview completion rate
    icr = signals.get("interview_completion_rate", 0) or 0
    score += icr * 0.10

    return min(score, 1.0)


def _score_location(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    location = _norm(profile.get("location", ""))
    signals = candidate.get("redrob_signals", {})
    willing = signals.get("willing_to_relocate", False)

    if _contains_any(location, PREFERRED_LOCATIONS):
        return 1.0
    if willing:
        return 0.6
    return 0.0


def _anti_pattern_penalty(candidate: Dict[str, Any]) -> float:
    """Returns a value in [0, 0.4] to subtract from the final score."""
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    penalty = 0.0

    companies = [_norm(j.get("company", "")) for j in history]
    titles_all = " ".join(_norm(j.get("title", "")) for j in history)
    skills_text = " ".join(_norm(s.get("name", "")) for s in skills)
    full_text = skills_text + " " + titles_all

    # Pure consulting career
    if all(_contains_any(c, CONSULTING_FIRMS) for c in companies) and companies:
        penalty += 0.20

    # CV/Speech/Robotics without NLP exposure
    cv_hits = _count_matches(full_text, CV_ONLY_SKILLS)
    nlp_hits = _count_matches(full_text, CORE_SKILLS)
    if cv_hits >= 3 and nlp_hits <= 1:
        penalty += 0.15

    # Inactivity (> 6 months)
    days = _days_since(signals.get("last_active_date", ""))
    if days > 180:
        penalty += 0.10

    # Very low recruiter response rate (< 5%)
    resp_rate = signals.get("recruiter_response_rate", 0.5) or 0
    if resp_rate < 0.05:
        penalty += 0.05

    # Keyword stuffing: 15+ ML skills but no ML titles in career history
    ml_skill_count = _count_matches(skills_text, CORE_SKILLS)
    ml_title_count = _count_matches(titles_all, ML_TITLES)
    if ml_skill_count >= 15 and ml_title_count == 0:
        penalty += 0.15

    return min(penalty, 0.40)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

WEIGHTS = {
    "trajectory": 0.30,
    "technical":  0.25,
    "production": 0.20,
    "behavioral": 0.15,
    "location":   0.05,
    "llm":        0.00,   # will be overridden in rank.py when LLM data present
}

LLM_BLEND_WEIGHT = 0.35  # when LLM score available, reallocate 35% to LLM


def score_candidate(
    candidate: Dict[str, Any],
    llm_score: float = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (final_score, breakdown_dict).
    llm_score: float in [0,1] from precomputed Gemini evaluation, or None.
    """
    d = {
        "trajectory": _score_career_trajectory(candidate),
        "technical":  _score_technical_depth(candidate),
        "production": _score_production_signals(candidate),
        "behavioral": _score_behavioral(candidate),
        "location":   _score_location(candidate),
        "penalty":    _anti_pattern_penalty(candidate),
        "llm":        llm_score if llm_score is not None else 0.0,
    }

    if llm_score is not None:
        # Blend: 35% LLM + 65% weighted rule-based
        rule_score = (
            d["trajectory"] * 0.30 +
            d["technical"]  * 0.25 +
            d["production"] * 0.20 +
            d["behavioral"] * 0.15 +
            d["location"]   * 0.05 +
            d["llm"]        * 0.05  # tiny LLM share in rule weights
        )
        final = llm_score * LLM_BLEND_WEIGHT + rule_score * (1 - LLM_BLEND_WEIGHT)
    else:
        final = (
            d["trajectory"] * WEIGHTS["trajectory"] +
            d["technical"]  * WEIGHTS["technical"]  +
            d["production"] * WEIGHTS["production"]  +
            d["behavioral"] * WEIGHTS["behavioral"]  +
            d["location"]   * WEIGHTS["location"]
        )

    # Apply anti-pattern penalty
    final = max(0.0, final - d["penalty"])

    return round(final, 6), d


def generate_rule_reasoning(candidate: Dict[str, Any], breakdown: Dict[str, float]) -> str:
    """Generates a 1-2 sentence factual reasoning using profile data."""
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    name = p.get("current_title", "Engineer")
    company = p.get("current_company", "current company")
    yoe = p.get("years_of_experience", 0)
    notice = s.get("notice_period_days", 90)
    location = p.get("location", "India")
    github = s.get("github_activity_score", -1)
    assessments = s.get("skill_assessment_scores") or {}

    # Pick top technical skills mentioned
    top_skills = [sk["name"] for sk in skills[:4]]
    skill_str = ", ".join(top_skills) if top_skills else "varied ML skills"

    # Current job context
    recent_job = history[0] if history else {}
    recent_title = recent_job.get("title", name)
    recent_co = recent_job.get("company", company)

    parts = [
        f"{recent_title} at {recent_co} with {yoe:.1f} years of experience; "
        f"key skills include {skill_str}."
    ]

    extras = []
    if github > 0:
        extras.append(f"GitHub activity score {github:.0f}/100")
    if assessments:
        best = max(assessments, key=assessments.get)
        extras.append(f"{best} assessment {assessments[best]:.0f}/100")
    if notice <= 30:
        extras.append("available within 30 days")
    if extras:
        parts.append(" ".join(extras).capitalize() + ".")

    return " ".join(parts)[:500]

