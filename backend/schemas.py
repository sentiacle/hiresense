"""
HireSense AI — Pydantic Schemas for API

Bug fixes:
  • CVAnalysisRequest now accepts jd_text (job description) so required_skills
    can be auto-extracted server-side — previously jd_text was never sent,
    leaving required_skills always [] which caused skills to be scored at 35/100
    regardless of JD match.
  • Weight keys are documented with BOTH plural and singular forms so callers
    know either is accepted (the scoring engine normalises them internally).
  • PredictSuccessResponse.combined_score field added (was missing from response).
  • Entity.confidence made truly optional (no default causes validation errors
    when the NER model doesn't emit confidence values).
"""

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator


class Entity(BaseModel):
    """A single extracted named entity."""
    text:       str   = Field(..., description="Entity text")
    label:      str   = Field(..., description="Entity type (SKILL, EXP, EDU, PROJ, ACH, CERT, ORG, LOC, DATE, NAME, CONTACT)")
    start:      int   = Field(..., description="Start token index")
    end:        int   = Field(..., description="End token index")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0,
                              description="Confidence score (0–1). Defaults to 1.0 if not provided by model.")


class CategoryScore(BaseModel):
    """Score and detail for one scoring category."""
    score:    float      = Field(..., ge=0, le=100, description="Category score (0–100)")
    weight:   float      = Field(..., ge=0,         description="Weight used (not necessarily out of 100 before normalisation)")
    entities: List[str]  = Field(default_factory=list, description="Entities contributing to this score")
    details:  str        = Field(default="",            description="Human-readable scoring explanation")


class CVAnalysisRequest(BaseModel):
    """
    Request payload for full CV analysis.

    Weights accept EITHER plural ('skills', 'projects', 'achievements')
    OR singular ('skill', 'project', 'achievement') keys — both work.
    Values do NOT need to sum to exactly 100; they are normalised internally.
    """
    cv_text:         str             = Field(..., min_length=10,
                                            description="Full text of the candidate's CV")
    jd_text:         Optional[str]   = Field(None,
                                            description="Raw job description text. STRONGLY RECOMMENDED — "
                                                        "omitting this causes skills to be scored without JD context. "
                                                        "When provided, required_skills is auto-extracted if not supplied.")
    job_id:          Optional[str]   = Field(None, description="Job ID for record-keeping")
    required_skills: List[str]       = Field(
        default_factory=list,
        description="Skills the job requires. Leave empty to auto-extract from jd_text.",
    )
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "skills":       40,
            "experience":   25,
            "projects":     15,
            "achievements": 10,
            "education":    10,
        },
        description=(
            "Category weights. Accepts plural or singular keys. "
            "Values are normalised to sum to 100 internally. "
            "Allowed keys: skills/skill, experience, projects/project, "
            "achievements/achievement, education."
        ),
    )

    @validator("weights")
    def validate_weights(cls, v):
        allowed = {
            "skills","skill","experience","education",
            "projects","project","achievements","achievement",
        }
        unknown = set(v.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown weight keys: {unknown}. Allowed: {allowed}")
        if any(val < 0 for val in v.values()):
            raise ValueError("All weight values must be non-negative.")
        return v

    @validator("required_skills", always=True)
    def auto_extract_skills(cls, v, values):
        """
        If required_skills is empty but jd_text is provided,
        auto-extract skill-like tokens from the JD.
        This ensures the skill score is always JD-relative.
        """
        if v:
            return v  # caller already provided skills

        jd = values.get("jd_text") or ""
        if not jd:
            return v  # nothing to extract from

        import re
        jd_lower = jd.lower()

        # Look for skills after common JD section headers
        section_match = re.search(
            r"(skills?|requirements?|qualifications?|technologies?|tools?|"
            r"tech stack|must.have|nice.to.have)[:\s]+(.*?)(?=\n\n|\Z)",
            jd_lower, re.DOTALL | re.IGNORECASE,
        )
        raw = section_match.group(2) if section_match else jd_lower

        # Split on delimiters common in JDs
        candidates = []
        for chunk in re.split(r"[,;•\-\|/\n]", raw):
            chunk = chunk.strip()
            for sub in re.split(r"\band\b|\bor\b|\+", chunk):
                sub = sub.strip()
                if 2 < len(sub) <= 35:
                    candidates.append(sub)

        # Remove obvious stop words
        STOP = {"the","a","an","to","of","for","in","with","on","at",
                "is","be","and","or","we","you","our","your","will",
                "can","may","should","experience","knowledge","strong"}
        extracted = [
            c for c in candidates
            if c.split()[0] not in STOP and re.search(r"[a-z]", c)
        ]

        return list(dict.fromkeys(extracted))[:30]  # deduplicate, cap at 30


class CVAnalysisResponse(BaseModel):
    """Response from CV analysis endpoint."""
    success:          bool                      = True
    entities:         List[Entity]              = Field(default_factory=list)
    scores:           Dict[str, CategoryScore]  = Field(default_factory=dict)
    total_score:      float                     = Field(..., ge=0, le=100)
    breakdown:        Dict[str, List[str]]       = Field(default_factory=dict)
    model_info:       Dict[str, str]            = Field(default_factory=dict)
    # FIX: expose whether JD was used and which skills were extracted from it
    jd_used:          bool                      = Field(default=False,
                                                        description="True if jd_text was provided and used for skill matching")
    skills_from_jd:   List[str]                 = Field(default_factory=list,
                                                        description="Skills auto-extracted from jd_text (empty if caller supplied required_skills)")


class EntityExtractionRequest(BaseModel):
    """Request raw entity extraction without scoring."""
    text: str = Field(..., min_length=1, description="Text to extract entities from")


class EntityExtractionResponse(BaseModel):
    """Response with extracted entities."""
    success:  bool        = True
    entities: List[Entity] = Field(default_factory=list)
    tokens:   List[str]   = Field(default_factory=list, description="Input tokens")
    labels:   List[str]   = Field(default_factory=list, description="Predicted BIO labels")


class TestGradeRequest(BaseModel):
    """Request to grade a completed MCQ test."""
    answers:         List[int] = Field(..., description="Candidate's answer indices (0-based)")
    correct_answers: List[int] = Field(..., description="Correct answer indices (0-based)")

    @validator("answers")
    def validate_answers_not_empty(cls, v):
        if not v:
            raise ValueError("answers must not be empty")
        return v


class TestGradeResponse(BaseModel):
    """Test grading result."""
    success: bool       = True
    score:   float      = Field(..., ge=0, le=100, description="Test score (0–100)")
    correct: int        = Field(..., description="Number of correct answers")
    total:   int        = Field(..., description="Total questions")
    details: List[Dict] = Field(default_factory=list, description="Per-question result")


class PredictSuccessRequest(BaseModel):
    """Request to predict hiring probability."""
    cv_score:       float = Field(..., ge=0, le=100, description="CV analysis score (0–100)")
    test_score:     float = Field(..., ge=0, le=100, description="MCQ test score (0–100)")
    min_cv_score:   float = Field(default=60,  ge=0, le=100, description="Minimum passing CV score")
    min_test_score: float = Field(default=50,  ge=0, le=100, description="Minimum passing test score")
    cv_weight:      float = Field(default=0.6, ge=0, le=1,   description="Weight given to CV score vs test score")


class PredictSuccessResponse(BaseModel):
    """Hiring probability prediction."""
    success:         bool  = True
    probability:     float = Field(..., ge=0, le=100, description="Estimated hire probability (%)")
    passed:          bool  = Field(..., description="True if candidate meets minimum score thresholds")
    combined_score:  float = Field(..., ge=0, le=100, description="Weighted combined score (cv_weight × cv + (1-cv_weight) × test)")


class HealthResponse(BaseModel):
    """Health check."""
    status:       str  = "ok"
    model_loaded: bool = False
    model_name:   str  = ""
    device:       str  = ""
