"""
honeypot.py - Identifies candidates with logically impossible profiles.

Detection rules (validated against sample_submission.csv: 0 false positives):
  R1 - Job duration_months > actual date span + 3 months (e.g. 8 yrs at company founded 3 yrs ago)
  R2 - >= 3 expert-level skills with duration_months == 0
  R3 - Headline years-of-experience differs from profile.years_of_experience by > 2
  R4 - Summary years-of-experience differs from profile.years_of_experience by > 2
  R5 - Profile years_of_experience > 3 but total career history < 1 year
  R6 - Declared years_of_experience exceeds career span (from earliest job) by > 3 years
  R7 - Sum of all job duration_months exceeds declared years_of_experience * 12 by > 48 months
"""


import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

_CURRENT_DATE = datetime(2026, 6, 22)

_YRS_RE = re.compile(
    r"(\d+\.?\d*)\+\s*(?:yrs|years)\s+(?:of\s+)?experience"
    r"|with\s+(\d+\.?\d*)\+\s*(?:yrs|years)",
    re.IGNORECASE,
)


def _parse_date(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _extract_years(text: str):
    if not text:
        return None
    m = _YRS_RE.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def is_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Returns (True, reasons_list) if candidate looks like a honeypot,
    (False, []) otherwise.
    """
    reasons: List[str] = []
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0) or 0
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # R1 - duration exceeds actual timeline
    for job in history:
        start = _parse_date(job.get("start_date"))
        dur = job.get("duration_months")
        if start is None or dur is None:
            continue
        end = _parse_date(job.get("end_date")) or _CURRENT_DATE
        diff = (end.year - start.year) * 12 + (end.month - start.month)
        if dur > diff + 3:
            reasons.append(
                f"R1:duration_mismatch {job.get('company')!r} "
                f"claims {dur}mo but dates imply {diff}mo"
            )
            break  # one job is enough to flag

    # R2 - expert skills with zero duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() == "expert"
        and s.get("duration_months", -1) == 0
    )
    if expert_zero >= 3:
        reasons.append(f"R2:expert_zero_duration x{expert_zero}")

    # R3 / R4 - experience claims in free text vs profile field
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    for label, text in [("R3:headline", headline), ("R4:summary", summary)]:
        y = _extract_years(text)
        if y is not None and abs(years_exp - y) > 2.0:
            reasons.append(f"{label}_exp_mismatch text={y} profile={years_exp}")

    # R5 - declared experience >> work history
    total_months = sum(j.get("duration_months", 0) or 0 for j in history)
    if years_exp > 3 and total_months < 12:
        reasons.append(
            f"R5:history_gap declared={years_exp}yrs history={total_months/12:.1f}yrs"
        )

    # R6 - declared YoE exceeds career span from earliest job by > 3 years
    starts = [_parse_date(j.get("start_date")) for j in history]
    starts = [s for s in starts if s is not None]
    if starts and years_exp > 0:
        earliest = min(starts)
        career_span_yrs = (_CURRENT_DATE - earliest).days / 365.25
        if years_exp > career_span_yrs + 3.0:
            reasons.append(
                f"R6:yoe_exceeds_career_span "
                f"declared={years_exp}yrs earliest_job={earliest.date()} "
                f"span={career_span_yrs:.1f}yrs"
            )

    # R7 - total history months >> declared YoE (impossible without time travel)
    if years_exp > 0 and total_months > (years_exp * 12 + 48):
        reasons.append(
            f"R7:history_exceeds_yoe "
            f"history={total_months/12:.1f}yrs declared={years_exp}yrs"
        )

    return bool(reasons), reasons

