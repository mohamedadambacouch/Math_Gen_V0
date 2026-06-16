"""
Platform Question Generator — Streamlit UI
Shares all generation logic with the API via core.py

Run from the Math_generator root:
    streamlit run generate_for_platform/app.py
"""

import json
import sys
from pathlib import Path
from typing import List

import streamlit as st

# Allow running as a script AND as a package member
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_for_platform.core import (
    count_existing,
    generate_questions,
    get_curriculum,
    get_subtopics,
    subtopics_for,
    topics_for_year,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Platform Question Generator",
    page_icon="📐",
    layout="wide",
)
st.title("📐 Platform Question Generator")
st.caption("Year 3–5 · UK national curriculum · qwen2-math:7b")

curriculum = get_curriculum()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    year = st.selectbox("Year group", ["3", "4", "5"], format_func=lambda y: f"Year {y}")

    topic = st.selectbox("Topic", topics_for_year(year))

    available_subs: List[str] = subtopics_for(year, topic)
    if available_subs:
        chosen_subs = st.multiselect(
            "Subtopics",
            available_subs,
            default=[],
            help="Leave empty to cover all subtopics.",
        )
    else:
        chosen_subs = []
        st.caption("No subtopics defined for this topic.")

    difficulty = st.select_slider(
        "Difficulty", options=["Easy", "Medium", "Hard"], value="Medium"
    )
    n_questions = st.slider("Number of questions", min_value=1, max_value=20, value=5)

    existing_count = count_existing(year, topic, difficulty)
    st.info(
        f"📚 **{existing_count}** existing *{difficulty}* question(s) in the bank "
        f"for this topic — used as style examples."
    )

    st.divider()
    generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

# ── Generation ─────────────────────────────────────────────────────────────────
if generate_btn:
    effective_subs = chosen_subs if chosen_subs else available_subs
    with st.spinner(
        f"Generating {n_questions} **{difficulty}** question(s) on *{topic}* "
        f"(Year {year})…"
    ):
        try:
            result = generate_questions(
                year=year,
                topic=topic,
                subtopics=effective_subs,
                difficulty=difficulty,
                n=n_questions,
            )
            qs       = result["questions"]
            attempts = result["attempts"]
            raw      = result["raw"]
        except Exception as e:
            st.error(f"Ollama error: {e}")
            st.info("Is Ollama running?  Try: `ollama serve`")
            qs, attempts, raw = [], 0, ""

    st.session_state.update(
        qs=qs, raw=raw, attempts=attempts,
        meta_year=year, meta_topic=topic,
        meta_diff=difficulty, meta_n=n_questions,
    )

# ── Results ────────────────────────────────────────────────────────────────────
qs       = st.session_state.get("qs") or []
raw      = st.session_state.get("raw", "")
attempts = st.session_state.get("attempts", 1)

if qs:
    got    = len(qs)
    wanted = st.session_state.get("meta_n", got)

    if got < wanted:
        st.warning(f"Got {got}/{wanted} questions after {attempts} attempt(s). Try again.")
    else:
        st.success(
            f"✅ {got} **{st.session_state.get('meta_diff')}** question(s) · "
            f"Year {st.session_state.get('meta_year')} · "
            f"{st.session_state.get('meta_topic')} · "
            f"{attempts} generation attempt(s)"
        )

    for i, q in enumerate(qs, 1):
        with st.container():
            col_num, col_body = st.columns([0.05, 0.95])
            with col_num:
                st.markdown(f"### {i}.")
            with col_body:
                st.markdown(f"**{q.get('question', '')}**")
                opts = q.get("options") or []
                for j, opt in enumerate(opts):
                    label = "ABCD"[j] if j < 4 else str(j + 1)
                    st.markdown(f"&nbsp;&nbsp;**{label}.** {opt}")
                with st.expander("✅ Show answer"):
                    st.markdown(f"**Correct answer:** {q.get('answer', '—')}")
                    expl = q.get("explanation", "")
                    if expl:
                        st.markdown(f"**Working:** {expl}")
        st.divider()

    yr_l   = st.session_state.get("meta_year", "")
    diff_l = st.session_state.get("meta_diff", "").lower()
    topic_l = (st.session_state.get("meta_topic") or "")[:30].replace(" ", "_").replace("/", "")
    st.download_button(
        "⬇️ Download as JSON",
        data=json.dumps(qs, indent=2, ensure_ascii=False),
        file_name=f"year{yr_l}_{diff_l}_{topic_l}.json",
        mime="application/json",
    )

elif generate_btn:
    st.error("❌ No questions returned. Check the debug panel and try again.")

if raw:
    with st.expander("🔍 Raw model output (debug)", expanded=not qs):
        st.code(raw, language="text")
