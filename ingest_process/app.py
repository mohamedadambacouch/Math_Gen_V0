"""
ingest_process/app.py
─────────────────────
4-phase pipeline:

  1. EXTRACT   — upload images / PDFs → MinerU + Qwen2-VL
  2. VALIDATE  — approve / reject each question, set difficulty
  3. LINK      — semantic topic matching against curriculum_345.json
  4. SAVE      — write approved+linked questions into questions_3/4/5_year.json

Run from the Math_generator root:
    streamlit run ingest_process/app.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest_process.extract import (
    MINERU_MODEL_ID,
    QWEN_MODEL_ID,
    load_existing,
    parse_mineru_output,
)
from extraction_pipeline import add_to_knowledge_base

CURRICULUM_FILE = ROOT / "curriculum_345.json"
OUTPUT_DEFAULT  = str(ROOT / "final_dataset.json")
VALID_YEARS     = ["3", "4", "5"]


# ── Cached model loaders ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_extraction_models():
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from mineru_vl_utils import MinerUClient

    proc_m  = AutoProcessor.from_pretrained(MINERU_MODEL_ID)
    model_m = Qwen2VLForConditionalGeneration.from_pretrained(
        MINERU_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    client = MinerUClient(model=model_m, processor=proc_m, backend="transformers")

    proc_q  = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model_q = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    return client, proc_q, model_q


@st.cache_resource(show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


@st.cache_data
def load_curriculum():
    with open(CURRICULUM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Helpers ────────────────────────────────────────────────────────────────────
def pdf_bytes_to_images(data: bytes, dpi: int) -> List[Image.Image]:
    import fitz
    doc   = fitz.open(stream=data, filetype="pdf")
    zoom  = dpi / 72
    mat   = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return pages


def uploaded_to_images(files, dpi: int) -> List[Dict]:
    items = []
    for f in files:
        ext = Path(f.name).suffix.lower()
        if ext == ".pdf":
            pages = pdf_bytes_to_images(f.read(), dpi)
            for i, img in enumerate(pages, 1):
                items.append({"name": f"{Path(f.name).stem}_page{i:03d}", "image": img})
        else:
            items.append({"name": Path(f.name).stem,
                          "image": Image.open(f).convert("RGB")})
    return items


def clean_text_for_embedding(text: str) -> str:
    text = re.sub(r"\d+\.", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def confidence_badge(score: float) -> str:
    if score >= 0.55:
        return "🟢 HIGH"
    if score >= 0.40:
        return "🟡 MEDIUM"
    return "🔴 LOW"


def strip_choice_letter(choice: str) -> str:
    """'A. Some text' → 'Some text' for storage in questions_X_year.json."""
    m = re.match(r"^[A-E]\.\s*(.*)", choice)
    return m.group(1).strip() if m else choice


def build_topic_options(curriculum: Dict) -> List[str]:
    """Flat list 'Year X — Topic' for override dropdowns."""
    opts = []
    for yr in VALID_YEARS:
        for t in curriculum["year_groups"][yr]["topics"]:
            opts.append(f"Year {yr} — {t}")
    return opts


def link_questions(questions: List[Dict], curriculum: Dict) -> List[Dict]:
    """Semantic match each question to the best curriculum topic."""
    embedder = load_embedder()

    all_topics = [
        {"year": yr, "topic": t}
        for yr in VALID_YEARS
        for t in curriculum["year_groups"][yr]["topics"]
    ]
    topic_texts      = [t["topic"] for t in all_topics]
    topic_embeddings = embedder.encode(topic_texts, show_progress_bar=False)

    from sklearn.metrics.pairwise import cosine_similarity

    linked = []
    for q in questions:
        clean = clean_text_for_embedding(q["question"])
        q_emb = embedder.encode([clean])
        sims  = cosine_similarity(q_emb, topic_embeddings)[0]
        best  = int(sims.argmax())
        score = float(sims[best])
        linked.append({
            **q,
            "topic":       all_topics[best]["topic"],
            "year":        all_topics[best]["year"],
            "topic_score": round(score, 4),
            "confidence":  confidence_badge(score),
        })
    return linked


def _reset():
    for key in ["phase", "extracted", "approved", "linked", "save_results"]:
        st.session_state.pop(key, None)


# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ingest & Extract", page_icon="🔬", layout="wide")
st.title("🔬 Exam Paper Ingest & Extract")

phase = st.session_state.get("phase", "extract")

# Phase indicator
PHASES = ["1 · Extract", "2 · Validate", "3 · Link Topics", "4 · Save"]
PHASE_MAP = {"extract": 0, "validate": 1, "link": 2, "save": 3}
step_cols = st.columns(4)
for i, (col, label) in enumerate(zip(step_cols, PHASES)):
    active = i == PHASE_MAP.get(phase, 0)
    col.markdown(
        f"<div style='text-align:center; padding:6px; border-radius:6px; "
        f"background:{'#1f77b4' if active else '#e0e0e0'}; "
        f"color:{'white' if active else '#555'}'><b>{label}</b></div>",
        unsafe_allow_html=True,
    )
st.divider()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader(
        "Images or PDFs",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"],
        accept_multiple_files=True,
        disabled=(phase != "extract"),
    )
    dpi = st.slider("PDF DPI", 100, 400, 200, 50)
    output_path = st.text_input("Save extracted JSON to", value=OUTPUT_DEFAULT)
    st.divider()

    if st.button("⚡ Load Models", use_container_width=True):
        with st.spinner("Loading MinerU + Qwen2-VL…"):
            load_extraction_models()
        st.success("✅ Extraction models ready")

    if st.button("🔗 Load Embedding Model", use_container_width=True):
        with st.spinner("Loading sentence-transformer…"):
            load_embedder()
        st.success("✅ Embedding model ready")

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        _reset()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
if phase == "extract":
    st.subheader("Step 1 — Upload & Extract")

    if uploaded:
        cols = st.columns(min(len(uploaded), 4))
        for i, f in enumerate(uploaded):
            with cols[i % 4]:
                if Path(f.name).suffix.lower() == ".pdf":
                    st.markdown(f"📄 **{f.name}**")
                    st.caption(f"{f.size // 1024} KB · PDF")
                else:
                    st.image(Image.open(f), caption=f.name, use_column_width=True)
                f.seek(0)

        if st.button("🚀 Extract Questions", type="primary"):
            with st.spinner("Loading models…"):
                try:
                    client, qwen_proc, qwen_model = load_extraction_models()
                except Exception as e:
                    st.error(f"Model load error: {e}")
                    st.stop()

            for f in uploaded:
                f.seek(0)

            items    = uploaded_to_images(uploaded, dpi)
            existing = load_existing(Path(output_path))

            st.subheader(f"Processing {len(items)} page(s)…")
            progress = st.progress(0.0)
            log_box  = st.empty()
            log_lines: List[str] = []

            def log(msg: str):
                log_lines.append(msg)
                log_box.code("\n".join(log_lines[-25:]), language="text")

            new_questions: List[Dict] = []
            skipped = 0

            for idx, item in enumerate(items, 1):
                log(f"[{idx}/{len(items)}] {item['name']}")
                try:
                    raw = client.two_step_extract(image=item["image"])
                except Exception as e:
                    log(f"  ❌ MinerU: {e}")
                    progress.progress(idx / len(items))
                    continue

                parsed = parse_mineru_output(raw, item["image"], qwen_proc, qwen_model)
                log(f"  found {len(parsed)} question(s)")

                for q in parsed:
                    qt = q["question"].strip()
                    if not qt:
                        continue
                    if qt in existing:
                        skipped += 1
                        log(f"  ↩ duplicate: {qt[:55]}")
                    else:
                        existing[qt] = q
                        new_questions.append(q)
                        log(f"  ✅ Q{q['question_id']}: {qt[:55]}")

                progress.progress(idx / len(items))

            # save raw extraction
            all_q = list(existing.values())
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_q, f, indent=2, ensure_ascii=False)
                log(f"\n💾 Saved {len(all_q)} total → {output_path}")
            except Exception as e:
                log(f"\n❌ Save error: {e}")

            progress.progress(1.0)

            if new_questions:
                st.success(f"✅ {len(new_questions)} new question(s) extracted  |  {skipped} duplicate(s) skipped")
                st.session_state["extracted"] = new_questions
                st.session_state["phase"]     = "validate"
                st.rerun()
            else:
                st.warning("No new questions found.")
    else:
        st.info("Upload one or more images or PDF files in the sidebar, then click **Extract Questions**.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
elif phase == "validate":
    extracted: List[Dict] = st.session_state.get("extracted", [])
    st.subheader(f"Step 2 — Validate {len(extracted)} Extracted Question(s)")
    st.caption("Approve the questions you want to keep, set each difficulty, then continue.")

    with st.form("validate_form"):
        form_data = []
        for i, q in enumerate(extracted):
            has_img = q.get("has_image", False)
            badge   = "🖼️" if has_img else "📝"

            with st.expander(f"{badge} Q{q['question_id']} — {q['question'][:80]}", expanded=True):
                col_check, col_diff = st.columns([3, 1])
                with col_check:
                    include = st.checkbox("✅ Approve", value=True, key=f"inc_{i}")
                with col_diff:
                    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"],
                                        index=1, key=f"diff_{i}")

                edited_q = st.text_area("Question text", value=q["question"],
                                        key=f"qtxt_{i}", height=80)
                choices  = q.get("choices", [])
                if choices:
                    st.markdown("**Choices:** " + " · ".join(choices))
                else:
                    st.caption("No choices extracted.")

                if has_img and q.get("image_description"):
                    with st.expander("Image description"):
                        st.info(q["image_description"])

                form_data.append({"q": q, "edited_q": edited_q, "include": include, "diff": diff})

        submitted = st.form_submit_button("➡️ Continue to Topic Linking", type="primary")

    if submitted:
        approved = []
        for fd in form_data:
            if fd["include"]:
                record = {**fd["q"], "question": fd["edited_q"].strip(), "difficulty": fd["diff"]}
                approved.append(record)

        if not approved:
            st.warning("No questions approved. Tick at least one.")
        else:
            st.session_state["approved"] = approved
            st.session_state["phase"]    = "link"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — LINK TOPICS
# ══════════════════════════════════════════════════════════════════════════════
elif phase == "link":
    approved: List[Dict] = st.session_state.get("approved", [])
    st.subheader(f"Step 3 — Link {len(approved)} Question(s) to Curriculum Topics")
    st.caption("The model suggests the best topic + year. Override any that look wrong.")

    curriculum = load_curriculum()
    topic_options = build_topic_options(curriculum)

    if "linked" not in st.session_state:
        with st.spinner("Running semantic topic matching…"):
            try:
                linked = link_questions(approved, curriculum)
                st.session_state["linked"] = linked
            except Exception as e:
                st.error(f"Linking error: {e}")
                st.stop()
        st.rerun()

    linked: List[Dict] = st.session_state["linked"]

    with st.form("link_form"):
        overrides = []
        for i, q in enumerate(linked):
            auto_label = f"Year {q['year']} — {q['topic']}"
            default_idx = topic_options.index(auto_label) if auto_label in topic_options else 0

            with st.expander(
                f"Q{q['question_id']} · {q['confidence']} · {auto_label[:60]} — {q['question'][:60]}",
                expanded=True,
            ):
                st.markdown(f"**Question:** {q['question']}")
                st.markdown(
                    f"**Auto match:** `{q['topic']}` · Year {q['year']} · "
                    f"score `{q['topic_score']}` · {q['confidence']}"
                )
                chosen = st.selectbox(
                    "Topic (override if wrong)",
                    topic_options,
                    index=default_idx,
                    key=f"topic_{i}",
                )
                overrides.append(chosen)

        submitted = st.form_submit_button("➡️ Save to Question Bank", type="primary")

    if submitted:
        # apply overrides
        final = []
        for q, override in zip(linked, overrides):
            yr_part, topic_part = override.split(" — ", 1)
            year = yr_part.replace("Year ", "").strip()
            final.append({**q, "topic": topic_part, "year": year})

        st.session_state["final_linked"] = final
        st.session_state["phase"]        = "save"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — SAVE
# ══════════════════════════════════════════════════════════════════════════════
elif phase == "save":
    final_linked: List[Dict] = st.session_state.get("final_linked", [])
    st.subheader(f"Step 4 — Saving {len(final_linked)} Question(s) to the Question Bank")

    if "save_results" not in st.session_state:
        results = []
        progress = st.progress(0.0)

        for idx, q in enumerate(final_linked, 1):
            question_text = q["question"].strip()
            topic         = q["topic"]
            year          = q["year"]
            difficulty    = q["difficulty"]
            options = [strip_choice_letter(c) for c in q.get("choices", [])] or None

            ok, msg = add_to_knowledge_base(
                question_text=question_text,
                topic=topic,
                year=year,
                difficulty=difficulty,
                options=options,
                base_dir=str(ROOT),
            )
            results.append({
                "question": question_text,
                "topic": topic,
                "year": year,
                "difficulty": difficulty,
                "ok": ok,
                "msg": msg,
            })
            progress.progress(idx / len(final_linked))

        st.session_state["save_results"] = results
        progress.progress(1.0)

    save_results = st.session_state["save_results"]

    saved   = [r for r in save_results if r["ok"]]
    dupes   = [r for r in save_results if not r["ok"] and r["msg"] == "duplicate"]
    errors  = [r for r in save_results if not r["ok"] and r["msg"] != "duplicate"]

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Saved",     len(saved))
    c2.metric("⚠️ Duplicates", len(dupes))
    c3.metric("❌ Errors",    len(errors))

    st.divider()
    for r in save_results:
        if r["ok"]:
            icon = "✅"
        elif r["msg"] == "duplicate":
            icon = "⚠️"
        else:
            icon = "❌"

        with st.expander(f"{icon} {r['question'][:80]}"):
            st.markdown(f"**Year:** {r['year']}  |  **Topic:** {r['topic']}  |  **Difficulty:** {r['difficulty']}")
            st.caption(f"Status: {r['msg']}")

    st.divider()
    if st.button("🔄 Process More Files", type="primary"):
        _reset()
        st.rerun()
