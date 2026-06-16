import streamlit as st
import json
import re
import os
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource(show_spinner=False)
def _load_mineru_client():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from mineru_vl_utils import MinerUClient
    import torch

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "opendatalab/MinerU2.5-2509-1.2B",
        dtype=torch.float16,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(
        "opendatalab/MinerU2.5-2509-1.2B",
        use_fast=True,
    )
    return MinerUClient(backend="transformers", model=model, processor=processor)


@st.cache_resource(show_spinner=False)
def _load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


def _group_into_questions(texts):
    """
    Reconstruct flat OCR text blocks into structured questions with options.

    Pattern:
      standalone number  → question separator  (e.g. "1", "2")
      option line        → A/B/C/D/E prefix    (e.g. "A 12", "B 11")
      everything else    → part of question text
    """
    questions = []
    current_parts = []
    current_opts  = []
    in_question   = False

    def _flush():
        if in_question and current_parts:
            questions.append({
                "question": " ".join(current_parts).strip(),
                "options":  current_opts[:],
            })

    for text in texts:
        text = text.strip()
        if not text:
            continue

        # Standalone question number → flush previous, start new
        if re.match(r'^\d+\.?$', text):
            _flush()
            current_parts = []
            current_opts  = []
            in_question   = True
            continue

        # Answer option (A/B/C/D/E followed by content)
        if re.match(r'^[A-Ea-e]\s*[\d\w]', text):
            opt_val = re.sub(r'^[A-Ea-e]\s*', '', text).strip()
            current_opts.append(opt_val)
            continue

        # Regular text → part of question body
        if in_question:
            current_parts.append(text)
        else:
            # Text before first number separator — treat as first question
            in_question = True
            current_parts.append(text)

    _flush()
    return questions


def extract_all_text(pil_image, resize_size: int = 1024):
    """
    Extract text from an image with MinerU, then group blocks into
    structured questions with options.
    Returns list of dicts: {"question": str, "options": [str, ...]}
    """
    client = _load_mineru_client()

    # Keep aspect ratio, just cap longest side
    img = pil_image.convert("RGB")
    w, h = img.size
    scale = resize_size / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    layout_data = client.two_step_extract(img)

    if not layout_data:
        return []

    raw_texts = []
    for item in layout_data:
        # Accept any dict-like or string result
        if isinstance(item, str):
            if item.strip():
                raw_texts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") in ["image", "footer", "page_number"]:
            continue
        # Try every common field name MinerU might use
        content = (
            item.get("content")
            or item.get("text")
            or item.get("value")
            or item.get("ocr_text")
            or ""
        ).strip()
        if content:
            raw_texts.append(content)

    return _group_into_questions(raw_texts)


def link_questions_to_topics(questions, curriculum_data):
    model = _load_sentence_model()

    all_topics = []
    for year, data in curriculum_data["year_groups"].items():
        for topic in data["topics"]:
            all_topics.append({"year": year, "topic": topic})

    topic_texts = [t["topic"] for t in all_topics]
    topic_embeddings = model.encode(topic_texts)

    results = []
    for q in questions:
        text = q.get("question", "").strip()
        if not text:
            continue
        clean = re.sub(r"\d+\.", "", text)
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        q_emb = model.encode([clean])
        sims = cosine_similarity(q_emb, topic_embeddings)[0]
        top_idx = sims.argsort()[-3:][::-1]

        matches = [
            {
                "topic": all_topics[i]["topic"],
                "year":  all_topics[i]["year"],
                "score": float(sims[i]),
            }
            for i in top_idx
        ]
        results.append({"question": text, "options": q.get("options", []), "matches": matches})

    return results


def add_to_knowledge_base(
    question_text: str,
    topic: str,
    year: str,
    difficulty: str,
    options: list = None,
    base_dir: str = ".",
):
    file_path = os.path.join(base_dir, f"questions_{year}_year.json")
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data.get("topics", []):
        if entry["topic"] == topic:
            examples = entry.setdefault("examples", {})
            bucket = examples.setdefault(difficulty, [])
            existing = {e["question"] for e in bucket}
            if question_text in existing:
                return False, "duplicate"
            record = {"question": question_text}
            if options:
                record["options"] = options
            bucket.append(record)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True, "added"

    return False, f"Topic not found in questions_{year}_year.json"
