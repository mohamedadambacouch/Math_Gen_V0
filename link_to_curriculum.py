"""
Link extracted questions (e.g. final_dataset.json) to the Year 3/4/5 curriculum
topics in curriculum_345.json, picking the single highest-scoring topic/year
match for each question.

This is step 1 of the pipeline:
  1. link_to_curriculum.py   -> waiting_for_llm_judge.json   (this script)
  2. judge_difficulty.py     -> waiting_for_llm_judge.json   (adds "difficulty": Easy/Medium/Hard)
  3. populate_question_bank.py -> questions_3/4/5_year.json  (final placement)

Usage:
    python link_to_curriculum.py [input_dataset.json]
"""

import json
import re
import sys
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CURRICULUM_FILE = "curriculum_345.json"
OUTPUT_FILE = "waiting_for_llm_judge.json"
QUESTION_BANK_FILES = {
    "3": "questions_3_year.json",
    "4": "questions_4_year.json",
    "5": "questions_5_year.json",
}


def clean_text(text: str) -> str:
    clean = re.sub(r"\d+\.", "", text)
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def confidence_label(score: float) -> str:
    if score >= 0.55:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def link_questions_batch(questions: List[Dict], curriculum_data: Dict) -> List[Dict]:
    """
    Semantic match a list of question dicts against curriculum_345 topics.
    Each input dict must have at least a 'question' key.
    Returns the same dicts with 'topic', 'year', 'topic_score', 'confidence' added.
    """
    model = SentenceTransformer("all-mpnet-base-v2")

    all_topics = []
    for year, data in curriculum_data["year_groups"].items():
        for topic in data["topics"]:
            all_topics.append({"year": year, "topic": topic})

    topic_embeddings = model.encode([t["topic"] for t in all_topics])

    linked = []
    for q in questions:
        text = q.get("question", "").strip()
        clean = clean_text(text)
        q_emb = model.encode([clean])
        sims  = cosine_similarity(q_emb, topic_embeddings)[0]
        best  = int(sims.argmax())
        score = float(sims[best])
        linked.append({
            **q,
            "topic":       all_topics[best]["topic"],
            "year":        all_topics[best]["year"],
            "topic_score": round(score, 4),
            "confidence":  confidence_label(score),
        })
    return linked


def load_existing_questions() -> set:
    """Question texts already present in the question banks or a previous
    waiting_for_llm_judge.json - skip these so re-runs don't duplicate."""
    seen = set()

    for path in QUESTION_BANK_FILES.values():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        for entry in data.get("topics", []):
            for bucket in entry.get("examples", {}).values():
                for item in bucket:
                    seen.add(item["question"].strip())

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for item in json.load(f):
                seen.add(item["question"].strip())
    except FileNotFoundError:
        pass

    return seen


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "final_dataset.json"

    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {input_file}")

    with open(CURRICULUM_FILE, "r", encoding="utf-8") as f:
        curriculum = json.load(f)

    all_topics = []
    for year, data in curriculum["year_groups"].items():
        for topic in data["topics"]:
            all_topics.append({"year": year, "topic": topic})
    print(f"Loaded {len(all_topics)} topics from {CURRICULUM_FILE}")

    already_seen = load_existing_questions()

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = []

    print("Loading embedding model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    topic_embeddings = model.encode([t["topic"] for t in all_topics])

    results: List[Dict[str, Any]] = []
    skipped_dupes = 0
    skipped_empty = 0

    for q in questions:
        question_text = q.get("question", "").strip()
        if not question_text:
            skipped_empty += 1
            continue
        if question_text in already_seen:
            skipped_dupes += 1
            continue

        q_embedding = model.encode([clean_text(question_text)])
        sims = cosine_similarity(q_embedding, topic_embeddings)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        result = {
            "question_id": q.get("question_id"),
            "question": question_text,
            "choices": q.get("choices", []),
            "has_image": q.get("has_image", False),
            "topic": all_topics[best_idx]["topic"],
            "year": all_topics[best_idx]["year"],
            "topic_score": round(best_score, 4),
            "confidence": confidence_label(best_score),
            "difficulty": None,
        }
        results.append(result)

        print(f"\nQ: {question_text[:80]}")
        print(f"  -> Year {result['year']} | {result['topic']} | score={result['topic_score']} ({result['confidence']})")

    merged = existing_results + results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nAdded {len(results)} new linked question(s) to {OUTPUT_FILE} (total now {len(merged)})")
    print(f"Skipped {skipped_dupes} duplicate(s) and {skipped_empty} empty question(s)")
    print("Next step: run the difficulty-judging script to fill in 'difficulty', "
          "then populate the questions_3/4/5_year.json files.")


if __name__ == "__main__":
    main()
