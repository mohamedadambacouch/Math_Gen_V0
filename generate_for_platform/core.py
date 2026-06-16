"""
Shared generation logic used by both the Streamlit app (app.py) and the
REST API (api.py).
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama

BASE = Path(__file__).resolve().parent.parent

CURRICULUM_FILE = BASE / "curriculum_345.json"
SUBTOPICS_FILE  = BASE / "subtopics.json"
QBANK_FILES = {
    "3": BASE / "questions_3_year.json",
    "4": BASE / "questions_4_year.json",
    "5": BASE / "questions_5_year.json",
}

MODEL    = "qwen2-math:7b"
NUM_CTX  = 8192
MAX_PRED = 6000

DIFFICULTY_GUIDE = {
    "Easy":   "single-step calculations, small friendly numbers, one skill tested at a time",
    "Medium": "two-step problems or moderately larger numbers, some reasoning required",
    "Hard":   "multi-step or abstract problems, large numbers, tricky applications or reasoning chains",
}

# ── Static data (loaded once) ──────────────────────────────────────────────────
def _load(path: Path, encoding: str = "utf-8") -> Any:
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


_curriculum: Optional[Dict] = None
_subtopics:  Optional[Dict] = None
_banks:      Dict[str, Dict] = {}


def get_curriculum() -> Dict:
    global _curriculum
    if _curriculum is None:
        _curriculum = _load(CURRICULUM_FILE)
    return _curriculum


def get_subtopics() -> Dict:
    global _subtopics
    if _subtopics is None:
        _subtopics = _load(SUBTOPICS_FILE, encoding="utf-8-sig")
    return _subtopics


def get_bank(year: str) -> Dict:
    if year not in _banks:
        _banks[year] = _load(QBANK_FILES[year])
    return _banks[year]


# ── Curriculum helpers ─────────────────────────────────────────────────────────
def topics_for_year(year: str) -> List[str]:
    return get_curriculum()["year_groups"][year]["topics"]


def subtopics_for(year: str, topic: str) -> List[str]:
    return (
        get_subtopics()
        .get("year_groups", {})
        .get(year, {})
        .get("topics", {})
        .get(topic, {})
        .get("subtopics", [])
    )


def count_existing(year: str, topic: str, difficulty: str) -> int:
    for entry in get_bank(year)["topics"]:
        if entry["topic"] == topic:
            return len(entry.get("examples", {}).get(difficulty, []))
    return 0


def get_existing(year: str, topic: str, difficulty: str, n: int = 4) -> List[Dict]:
    for entry in get_bank(year)["topics"]:
        if entry["topic"] == topic:
            pool = entry.get("examples", {}).get(difficulty, [])
            return random.sample(pool, min(n, len(pool)))
    return []


# ── Prompt ─────────────────────────────────────────────────────────────────────
_JSON_SCHEMA = """[
  {
    "question": "Full self-contained question text — no image or diagram needed.",
    "options": ["option A text", "option B text", "option C text", "option D text"],
    "answer": "exact text of the correct option",
    "explanation": "1-2 sentence step-by-step working"
  }
]"""


def _format_few_shot(examples: List[Dict]) -> str:
    if not examples:
        return "  (No existing examples yet for this combination.)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"  Example {i}:")
        lines.append(f"    Q: {ex['question']}")
        for j, opt in enumerate(ex.get("options", [])):
            lines.append(f"    {'ABCD'[j]}) {opt}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(
    year: str,
    topic: str,
    subtopics: List[str],
    difficulty: str,
    n: int,
    existing: List[Dict],
) -> str:
    subtopic_block = (
        "\n".join(f"    • {s}" for s in subtopics)
        if subtopics
        else "    (All subtopics for this topic)"
    )
    guide = DIFFICULTY_GUIDE[difficulty]
    few_shot = _format_few_shot(existing)

    return f"""You are an expert UK primary-school mathematics teacher creating exam questions for a digital learning platform.

YEAR GROUP : Year {year}
TOPIC      : {topic}
DIFFICULTY : {difficulty}  —  {guide}

SUBTOPICS TO COVER:
{subtopic_block}

EXISTING QUESTIONS IN THE BANK (style reference only — do NOT copy or closely rephrase):
{few_shot}

YOUR TASK — generate exactly {n} brand-new, original multiple-choice questions that:
  1. Are fully self-contained text (no images, graphs or diagrams needed).
  2. Each tests one of the subtopics listed above.
  3. Precisely match the {difficulty} difficulty level described above.
  4. Use completely different numbers, names and scenarios from the existing examples.
  5. Together cover as many of the listed subtopics as possible.
  6. Have exactly 4 distinct answer options — one correct, three plausible distractors.
  7. Are written in plain English with simple notation (e.g. 1/2, 30%, sqrt(9)) — NO LaTeX.

Return ONLY a valid JSON array (no markdown fences, no text outside the JSON) with exactly {n} objects:
{_JSON_SCHEMA}

Generate exactly {n} questions now as a pure JSON array:"""


# ── Robust JSON extraction ─────────────────────────────────────────────────────
def extract_json(text: str) -> List[Dict]:
    start = text.find("[")
    if start == -1:
        return []
    s = text[start:]

    balance, last_valid = 0, 0
    for i, ch in enumerate(s):
        if ch == "[":
            balance += 1
        elif ch == "]":
            balance -= 1
            if balance == 0:
                last_valid = i + 1
    if last_valid == 0:
        lc = s.rfind("]")
        if lc != -1:
            last_valid = lc + 1
    s = s[:last_valid]
    if balance > 0:
        s += "]" * balance

    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", s)
    s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\d?sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\times", "*", s)
    s = re.sub(r"\\div", "/", s)
    s = re.sub(r"\\cdot", "*", s)
    s = re.sub(r"\\circ", " degrees", s)
    s = re.sub(r"\\[,;]", " ", s)
    s = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

    def try_parse(x: str):
        try:
            d = json.loads(x)
            return d if isinstance(d, list) else None
        except json.JSONDecodeError:
            return None

    result = try_parse(s)
    if result is not None:
        return result

    lb = s.rfind("}")
    if lb != -1:
        t = re.sub(r",\s*]", "]", s[: lb + 1] + "]")
        result = try_parse(t)
        if result is not None:
            return result

    return []


# ── Main generation function ───────────────────────────────────────────────────
def generate_questions(
    year: str,
    topic: str,
    subtopics: List[str],
    difficulty: str,
    n: int,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Returns:
        {
            "questions": [...],
            "attempts":  int,
            "raw":       str   # last raw model output, useful for debugging
        }
    """
    existing = get_existing(year, topic, difficulty, n=4)
    prompt   = build_prompt(year, topic, subtopics, difficulty, n, existing)
    best: List[Dict] = []
    raw = ""

    for attempt in range(1, max_attempts + 1):
        resp = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={"temperature": 0.7, "num_predict": MAX_PRED, "num_ctx": NUM_CTX},
        )
        raw = resp.get("response", "")
        parsed = extract_json(raw)
        if len(parsed) >= n:
            return {"questions": parsed[:n], "attempts": attempt, "raw": raw}
        if len(parsed) > len(best):
            best = parsed

    return {"questions": best, "attempts": max_attempts, "raw": raw}
