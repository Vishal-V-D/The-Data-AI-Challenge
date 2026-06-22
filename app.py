"""
app.py - Streamlit dashboard for the Redrob Candidate Ranking solution.

Run:
    cd solution
    streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path resolution (must come before any local imports) ──────────────────────
_ROOT = Path(__file__).resolve().parent          # …/solution
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))               # makes `src` package importable

_DATA_DIR = (
    _ROOT.parent
    / "extracted_data"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
)
CANDIDATES_PATH = _DATA_DIR / "candidates.jsonl"
LLM_DATA_PATH   = _ROOT / "data" / "precomputed_llm_data.json"
OUTPUT_PATH      = _ROOT / "outputs" / "submission.csv"
VALIDATE_SCRIPT  = _DATA_DIR / "validate_submission.py"
# ─────────────────────────────────────────────────────────────────────────────

from src.honeypot import is_honeypot
from src.scorer   import score_candidate, generate_rule_reasoning

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling & Icon Pack (Material Icons Outlined) ──────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .score-high { color: #4ade80; font-weight: 600; }
  .score-mid  { color: #fb923c; font-weight: 600; }
  .score-low  { color: #f87171; font-weight: 600; }
  .stMetric label { font-size: 0.78rem; color: #888; }
  .material-icons-outlined {
    vertical-align: middle;
    line-height: 1;
  }
</style>
""", unsafe_allow_html=True)


def icon(name: str, color: str = "#6366f1", size: int = 20) -> str:
    """Generate HTML string for a material design icon."""
    return f'<span class="material-icons-outlined" style="color: {color}; font-size: {size}px; vertical-align: middle; margin-right: 6px;">{name}</span>'


# ── API key helpers ────────────────────────────────────────────────────────────

def _load_groq_key() -> str:
    for p in [_ROOT / ".env", _ROOT.parent / ".env"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k.strip() == "GROQ_API_KEY":
                        return v.strip()
    return ""


@st.cache_data(show_spinner=False)
def _validate_groq_key(api_key: str) -> tuple[bool, str]:
    if not api_key or not api_key.startswith("gsk_"):
        return False, "GROQ_API_KEY missing or wrong format (must start with gsk_)."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True, "Groq API Connection Established."
    except Exception as e:
        return False, f"API key error: {str(e)[:250]}"


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading candidates...")
def load_candidates(limit: int) -> list:
    if not CANDIDATES_PATH.exists():
        return []
    out = []
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@st.cache_data(show_spinner="Loading LLM data...")
def load_llm_data() -> dict:
    if LLM_DATA_PATH.exists():
        with open(LLM_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_specific_candidates(cids: set[str]) -> dict:
    if not CANDIDATES_PATH.exists():
        return {}
    out = {}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            if cid in cids:
                out[cid] = cand
                if len(out) == len(cids):
                    break
    return out


@st.cache_data(show_spinner="Scoring candidates...")
def run_scoring(limit: int) -> pd.DataFrame:
    candidates = load_candidates(limit)
    llm_data   = load_llm_data()
    rows = []
    for cand in candidates:
        flagged, _ = is_honeypot(cand)
        if flagged:
            continue
        cid = cand["candidate_id"]
        llm_entry  = llm_data.get(cid)
        llm_score  = llm_entry["llm_score"] if llm_entry else None
        try:
            final, breakdown = score_candidate(cand, llm_score)
        except Exception:
            continue
        p = cand.get("profile", {})
        s = cand.get("redrob_signals", {})
        
        # Reasoning fallback check
        llm_reasoning = llm_entry.get("reasoning", "").strip() if llm_entry else ""
        if llm_reasoning and "unavailable" not in llm_reasoning.lower():
            reasoning = llm_reasoning
        else:
            reasoning = generate_rule_reasoning(cand, breakdown)
            
        rows.append({
            "rank":         0,
            "candidate_id": cid,
            "score":        round(float(final), 4),
            "name":         p.get("anonymized_name", cid),
            "title":        p.get("current_title", ""),
            "company":      p.get("current_company", ""),
            "location":     p.get("location", ""),
            "yoe":          float(p.get("years_of_experience", 0) or 0),
            "notice_days":  int(s.get("notice_period_days", 90) or 90),
            "github":       float(s.get("github_activity_score", -1) or -1),
            "resp_rate":    float(s.get("recruiter_response_rate", 0) or 0),
            "last_active":  s.get("last_active_date", "") or "",
            "traj":         round(float(breakdown.get("trajectory", 0)), 3),
            "tech":         round(float(breakdown.get("technical", 0)), 3),
            "prod":         round(float(breakdown.get("production", 0)), 3),
            "behav":        round(float(breakdown.get("behavioral", 0)), 3),
            "loc":          round(float(breakdown.get("location", 0)), 3),
            "penalty":      round(float(breakdown.get("penalty", 0)), 3),
            "llm_score":    float(llm_score or 0),
            "reasoning":    reasoning,
        })
    if not rows:
        cols = ["rank","candidate_id","score","name","title","company","location",
                "yoe","notice_days","github","resp_rate","last_active",
                "traj","tech","prod","behav","loc","penalty","llm_score","reasoning"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"<h2>{icon('stars', '#6366f1', 28)} Redrob AI Ranker</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
analysis_limit = st.sidebar.slider("Candidates to analyse", 1_000, 50_000, 10_000, 1_000)
show_top       = st.sidebar.slider("Show top N in table", 10, 100, 25)
st.sidebar.markdown("---")

groq_key = _load_groq_key()
key_ok, key_msg = _validate_groq_key(groq_key)
if key_ok:
    st.sidebar.success(f"Groq API connection verified.")
else:
    st.sidebar.warning(key_msg)

# ── Main tabs ─────────────────────────────────────────────────────────────────
st.markdown(f"<h1>{icon('radar', '#6366f1', 36)} Candidate Discovery & Ranking</h1>", unsafe_allow_html=True)
st.markdown("**Redrob AI Challenge** — Multi-signal hybrid scoring engine")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Top Candidates",
    "Explorer", "Submission", "Evaluate Accuracy"
])

# ─── Tab 1: Dashboard ────────────────────────────────────────────────────────
with tab1:
    df      = run_scoring(analysis_limit)
    top100  = df.head(100)
    n       = len(df)
    n_hp    = analysis_limit - n
    top_s   = df["score"].max() if n > 0 else 0.0
    n_llm   = int((df["llm_score"] > 0).sum()) if n > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates Scored",  f"{n:,}")
    c2.metric("Honeypots Filtered", f"{n_hp:,}")
    c3.metric("Top Score",          f"{top_s:.4f}")
    c4.metric("LLM Evaluated",      f"{n_llm:,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Score Distribution (Top 500)")
        if n > 0:
            st.bar_chart(df.head(500)["score"].values)
    with col_b:
        st.subheader("Dimension Breakdown (Top 25 avg)")
        if n > 0:
            dims = df.head(25)[["traj","tech","prod","behav","loc","penalty"]].mean()
            st.bar_chart(dims)

    st.subheader("Location Distribution (Top 100)")
    if n > 0:
        st.bar_chart(top100["location"].value_counts().head(10))


# ─── Tab 2: Top Candidates ────────────────────────────────────────────────────
with tab2:
    df2    = run_scoring(analysis_limit)
    top_df = df2.head(show_top)

    for _, row in top_df.iterrows():
        label = (
            f"#{int(row['rank'])}  {row['name']}  —  "
            f"{row['title']} @ {row['company']}  "
            f"[{row['location']}] | Score: {row['score']:.4f}"
        )
        with st.expander(label, expanded=row["rank"] == 1):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Score",   f"{row['score']:.4f}")
            c2.metric("Experience",    f"{row['yoe']:.1f} yrs")
            c3.metric("Notice Period", f"{int(row['notice_days'])} days")
            c4.metric("GitHub",        f"{row['github']:.0f}" if row["github"] >= 0 else "—")

            cols = st.columns(6)
            for col, lbl, val in zip(cols,
                    ["Trajectory","Technical","Production","Behavioral","Location","Penalty"],
                    [row["traj"],row["tech"],row["prod"],row["behav"],row["loc"],row["penalty"]]):
                col.metric(lbl, f"{val:.2f}")

            if row["reasoning"]:
                st.info(f"**Reasoning:** {row['reasoning']}")


# ─── Tab 3: Explorer ──────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"### {icon('search', '#6366f1')} Search Candidates", unsafe_allow_html=True)
    df3    = run_scoring(analysis_limit)
    search = st.text_input("Search by name, title, or company")
    min_sc = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)

    filt = df3[df3["score"] >= min_sc]
    if search:
        mask = (
            filt["name"].str.contains(search, case=False, na=False)
            | filt["title"].str.contains(search, case=False, na=False)
            | filt["company"].str.contains(search, case=False, na=False)
        )
        filt = filt[mask]

    st.dataframe(
        filt[["rank","candidate_id","name","title","company","location",
              "yoe","score","notice_days","github"]].head(50),
        use_container_width=True,
    )


# ─── Tab 4: Submission ────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"### {icon('folder_open', '#6366f1')} Generate & Validate Submission", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Run Full Ranking", type="primary"):
            with st.spinner("Ranking 100,000 candidates..."):
                result = subprocess.run(
                    [sys.executable, str(_ROOT / "rank.py"), "--output", str(OUTPUT_PATH)],
                    capture_output=True, text=True, cwd=str(_ROOT)
                )
            if result.returncode == 0:
                st.success("Submission generated successfully!")
                st.code(result.stdout)
            else:
                st.error("Error generating submission.")
                st.code(result.stderr)

    with col_b:
        if OUTPUT_PATH.exists() and st.button("Validate Submission"):
            result = subprocess.run(
                [sys.executable, str(VALIDATE_SCRIPT), str(OUTPUT_PATH)],
                capture_output=True, text=True
            )
            if "valid" in result.stdout.lower():
                st.success(result.stdout.strip())
            else:
                st.error(result.stdout + result.stderr)

    if OUTPUT_PATH.exists():
        sub_df = pd.read_csv(OUTPUT_PATH)
        st.subheader(f"Current Submission — {len(sub_df)} candidates")
        st.dataframe(sub_df, use_container_width=True)
        with open(OUTPUT_PATH, "rb") as f:
            st.download_button("Download submission.csv", f,
                               file_name="submission.csv", mime="text/csv")


# ─── Tab 5: Evaluate Accuracy ────────────────────────────────────────────────
with tab5:
    st.markdown(f"## {icon('biotech', '#6366f1', 28)} Evaluate Reasoning & Ranking Quality", unsafe_allow_html=True)
    st.markdown("""
    This tab analyses the **quality of the top-100 submission** across 4 dimensions:
    - **Honeypot rate** — checks that no impossible profiles are present in the final shortlist
    - **Reasoning quality** — measures length, uniqueness, and format compliance of reasoning fields
    - **LLM coverage** — validates candidate evaluations against LLM data
    - **Score consistency** — validates non-increasing score order requirements
    """)

    if not OUTPUT_PATH.exists():
        st.warning("No submission.csv found. Go to the Submission tab and run ranking first.")
    else:
        sub_df    = pd.read_csv(OUTPUT_PATH)
        llm_data4 = load_llm_data()
        df4       = run_scoring(analysis_limit)

        st.markdown("---")

        # ── 1. Honeypot check ────────────────────────────────────────────
        st.markdown(f"### {icon('security', '#fb923c')} 1. Honeypot Check", unsafe_allow_html=True)
        
        cids_to_check = set(sub_df["candidate_id"])
        cand_map = load_specific_candidates(cids_to_check)
        
        hp_flags = []
        for cid in sub_df["candidate_id"]:
            cand = cand_map.get(cid)
            if cand:
                flagged, _ = is_honeypot(cand)
                if flagged:
                    hp_flags.append(cid)
            else:
                hp_flags.append(cid)

        hp_rate = len(hp_flags) / 100
        if hp_rate == 0:
            st.success("Honeypot rate: 0 / 100 (0.0%) — safe (threshold < 10%)")
        elif hp_rate < 0.10:
            st.warning(f"Honeypot rate: {len(hp_flags)} / 100 ({hp_rate*100:.1f}%) — within acceptable limit")
        else:
            st.error(f"Honeypot rate: {len(hp_flags)} / 100 ({hp_rate*100:.1f}%) — DISQUALIFIED (limit < 10%)")

        if hp_flags:
            st.write("Flagged Honeypot IDs:", hp_flags)

        # ── 2. Score monotonicity ─────────────────────────────────────────
        st.markdown(f"### {icon('trending_down', '#3b82f6')} 2. Score Monotonicity", unsafe_allow_html=True)
        scores   = sub_df["score"].tolist()
        viols    = [(i+1, i+2, scores[i], scores[i+1])
                    for i in range(len(scores)-1) if scores[i] < scores[i+1]]
        if not viols:
            st.success("Scores are non-increasing — spec compliant.")
        else:
            st.error(f"{len(viols)} monotonicity violation(s) detected:")
            for r1, r2, s1, s2 in viols[:10]:
                st.write(f"  Rank {r1} score={s1:.4f} < Rank {r2} score={s2:.4f}")

        # ── 3. Reasoning quality ──────────────────────────────────────────
        st.markdown(f"### {icon('description', '#10b981')} 3. Reasoning Quality", unsafe_allow_html=True)
        reasonings = sub_df["reasoning"].tolist()

        lengths     = [len(str(r)) for r in reasonings]
        unique_cnt  = len(set(reasonings))
        empty_cnt   = sum(1 for r in reasonings if not str(r).strip())
        short_cnt   = sum(1 for r in reasonings if 0 < len(str(r)) < 40)
        llm_cnt     = sum(1 for cid in sub_df["candidate_id"] if cid in llm_data4)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unique Reasonings",  f"{unique_cnt} / 100")
        col2.metric("Empty Reasonings",   f"{empty_cnt}")
        col3.metric("Short (<40 chars)",  f"{short_cnt}")
        col4.metric("LLM-backed",         f"{llm_cnt} / 100")

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        st.write(f"Average reasoning length: **{avg_len:.0f} characters**")

        if unique_cnt == 100 and empty_cnt == 0 and short_cnt == 0:
            st.success("All reasonings are unique, non-empty, and adequately long.")
        else:
            if unique_cnt < 100:
                st.error(f"{100 - unique_cnt} duplicate reasoning(s) detected.")
            if empty_cnt:
                st.error(f"{empty_cnt} empty reasoning(s) detected.")
            if short_cnt:
                st.warning(f"{short_cnt} reasoning(s) are very short (<40 chars).")

        st.markdown("#### Reasoning Length Distribution")
        st.bar_chart(pd.DataFrame({"length": lengths}, index=sub_df["rank"]))

        # ── 4. Dimension breakdown for top 100 ───────────────────────────
        st.markdown(f"### {icon('analytics', '#8b5cf6')} 4. Top-100 Scoring Dimension Breakdown", unsafe_allow_html=True)
        top100_ids = set(sub_df["candidate_id"])
        top100_rows = df4[df4["candidate_id"].isin(top100_ids)]
        if not top100_rows.empty:
            means = top100_rows[["traj","tech","prod","behav","loc","penalty"]].mean()
            col_left, col_right = st.columns(2)
            with col_left:
                st.bar_chart(means)
            with col_right:
                st.dataframe(
                    top100_rows[["rank","name","title","company","yoe","score",
                                  "traj","tech","prod","behav","loc","penalty"]]
                    .head(20),
                    use_container_width=True,
                )

        # ── 5. Sample reasonings ──────────────────────────────────────────
        st.markdown(f"### {icon('toc', '#6b7280')} 5. Sample Reasonings (Top 10)", unsafe_allow_html=True)
        for _, row in sub_df.head(10).iterrows():
            with st.expander(f"Rank {int(row['rank'])} — {row['candidate_id']}"):
                st.write(f"**Score:** {row['score']}")
                st.write(f"**Reasoning:** {row['reasoning']}")
                if row['candidate_id'] in llm_data4:
                    st.caption("AI-generated reasoning")
                else:
                    st.caption("Rule-based fallback reasoning")
