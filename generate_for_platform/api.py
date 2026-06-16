"""
REST API for the Platform Question Generator
Powered by FastAPI + qwen2-math:7b via Ollama

Start:
    uvicorn generate_for_platform.api:app --reload --port 8000

Interactive docs (auto-generated):
    http://localhost:8000/docs        ← Swagger UI
    http://localhost:8000/redoc       ← ReDoc
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .core import (
    count_existing,
    generate_questions,
    subtopics_for,
    topics_for_year,
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Platform Question Generator API",
    description=(
        "Generates original UK primary-school (Year 3–5) maths multiple-choice "
        "questions using a local qwen2-math:7b model via Ollama."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_YEARS       = {"3", "4", "5"}
VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}


# ── Pydantic models ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    year: str = Field(..., description="Year group: '3', '4' or '5'")
    topic: str = Field(..., description="Full topic name as it appears in the curriculum")
    subtopics: Optional[List[str]] = Field(
        default=None,
        description="Subtopics to focus on. Omit or pass [] to cover all subtopics.",
    )
    difficulty: str = Field(
        default="Medium",
        description="Difficulty level: 'Easy', 'Medium' or 'Hard'",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate (1–20)",
    )

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: str) -> str:
        if v not in VALID_YEARS:
            raise ValueError(f"year must be one of {sorted(VALID_YEARS)}")
        return v

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        if v not in VALID_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(VALID_DIFFICULTIES)}")
        return v


class Question(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: str = ""


class GenerateResponse(BaseModel):
    year: str
    topic: str
    difficulty: str
    requested: int
    generated: int
    attempts: int
    questions: List[Question]


class TopicInfo(BaseModel):
    topic: str
    subtopics: List[str]
    bank_counts: Dict[str, int]


class CurriculumResponse(BaseModel):
    year: str
    topics: List[TopicInfo]


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health() -> Dict[str, str]:
    """Quick liveness check."""
    return {"status": "ok"}


@app.get(
    "/curriculum/{year}",
    response_model=CurriculumResponse,
    tags=["Curriculum"],
    summary="List topics and subtopics for a year group",
)
def curriculum(year: str) -> CurriculumResponse:
    """
    Returns every topic for the requested year, its subtopics, and
    how many existing questions are in the bank per difficulty level.
    """
    if year not in VALID_YEARS:
        raise HTTPException(status_code=400, detail=f"year must be one of {sorted(VALID_YEARS)}")

    topics = topics_for_year(year)
    result = []
    for t in topics:
        result.append(
            TopicInfo(
                topic=t,
                subtopics=subtopics_for(year, t),
                bank_counts={
                    d: count_existing(year, t, d)
                    for d in ("Easy", "Medium", "Hard")
                },
            )
        )
    return CurriculumResponse(year=year, topics=result)


@app.get(
    "/subtopics",
    tags=["Curriculum"],
    summary="List subtopics for a specific year + topic",
)
def subtopics(
    year: str  = Query(..., description="Year group: '3', '4' or '5'"),
    topic: str = Query(..., description="Full topic name"),
) -> Dict[str, Any]:
    if year not in VALID_YEARS:
        raise HTTPException(status_code=400, detail=f"year must be one of {sorted(VALID_YEARS)}")
    subs = subtopics_for(year, topic)
    if not subs:
        raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found for Year {year}")
    return {"year": year, "topic": topic, "subtopics": subs}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["Generation"],
    summary="Generate original maths questions",
)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Generate `n` original multiple-choice questions for the requested
    year / topic / subtopics / difficulty combination.

    - **year** – `"3"`, `"4"` or `"5"`
    - **topic** – exact topic string from `/curriculum/{year}`
    - **subtopics** – list from `/subtopics`; omit to cover all
    - **difficulty** – `"Easy"`, `"Medium"` or `"Hard"`
    - **n** – 1–20 questions
    """
    # Validate topic exists for this year
    valid_topics = topics_for_year(req.year)
    if req.topic not in valid_topics:
        raise HTTPException(
            status_code=400,
            detail=f"Topic '{req.topic}' not found for Year {req.year}. "
                   f"Valid topics: {valid_topics}",
        )

    effective_subs = req.subtopics or subtopics_for(req.year, req.topic)

    try:
        result = generate_questions(
            year=req.year,
            topic=req.topic,
            subtopics=effective_subs,
            difficulty=req.difficulty,
            n=req.n,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model error: {exc}")

    raw_qs = result["questions"]
    # Normalise each question to the response schema
    questions = [
        Question(
            question=q.get("question", ""),
            options=q.get("options", []),
            answer=q.get("answer", ""),
            explanation=q.get("explanation", ""),
        )
        for q in raw_qs
        if q.get("question") and q.get("options")
    ]

    return GenerateResponse(
        year=req.year,
        topic=req.topic,
        difficulty=req.difficulty,
        requested=req.n,
        generated=len(questions),
        attempts=result["attempts"],
        questions=questions,
    )
