"""
HireSense AI - Pydantic Schemas for API
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Extracted named entity"""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity type (SKILL, EXP, EDU, etc.)")
    start: int = Field(..., description="Start token index")
    end: int = Field(..., description="End token index")
    confidence: float = Field(default=1.0, description="Confidence score")


class CategoryScore(BaseModel):
    """Score for a single category"""
    score: float = Field(..., ge=0, le=100, description="Score out of 100")
    weight: float = Field(..., ge=0, le=100, description="Weight percentage")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    details: str = Field(default="", description="Scoring details")


class CVAnalysisRequest(BaseModel):
    """Request to analyze a CV"""
    cv_text: str = Field(..., min_length=10, description="CV text content")
    job_id: Optional[str] = Field(None, description="Job ID for matching")
    required_skills: List[str] = Field(default_factory=list, description="Required skills to match")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "skills": 40,
            "experience": 25,
            "projects": 15,
            "achievements": 10,
            "education": 10
        },
        description="Category weights (must sum to 100)"
    )


class CVAnalysisResponse(BaseModel):
    """Response from CV analysis"""
    success: bool = True
    entities: List[Entity] = Field(default_factory=list, description="All extracted entities")
    scores: Dict[str, CategoryScore] = Field(default_factory=dict, description="Scores by category")
    total_score: float = Field(..., ge=0, le=100, description="Weighted total score")
    breakdown: Dict[str, List[str]] = Field(default_factory=dict, description="Entities grouped by type")
    model_info: Dict[str, str] = Field(default_factory=dict, description="Model metadata")


class EntityExtractionRequest(BaseModel):
    """Request for raw entity extraction"""
    text: str = Field(..., min_length=1, description="Text to extract entities from")


class EntityExtractionResponse(BaseModel):
    """Response with extracted entities"""
    success: bool = True
    entities: List[Entity] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list, description="Input tokens")
    labels: List[str] = Field(default_factory=list, description="Predicted labels")


class TestGradeRequest(BaseModel):
    """Request to grade a test"""
    answers: List[int] = Field(..., description="User's answer indices")
    correct_answers: List[int] = Field(..., description="Correct answer indices")


class TestGradeResponse(BaseModel):
    """Response with test grade"""
    success: bool = True
    score: float = Field(..., ge=0, le=100, description="Test score")
    correct: int = Field(..., description="Number of correct answers")
    total: int = Field(..., description="Total questions")
    details: List[Dict] = Field(default_factory=list, description="Per-question results")


class PredictSuccessRequest(BaseModel):
    """Request to predict hiring success"""
    cv_score: float = Field(..., ge=0, le=100)
    test_score: float = Field(..., ge=0, le=100)
    min_cv_score: float = Field(default=60, ge=0, le=100)
    min_test_score: float = Field(default=50, ge=0, le=100)
    cv_weight: float = Field(default=0.6, ge=0, le=1)


class PredictSuccessResponse(BaseModel):
    """Response with success prediction"""
    success: bool = True
    probability: float = Field(..., ge=0, le=100, description="Hire probability %")
    passed: bool = Field(..., description="Whether candidate passed criteria")
    combined_score: float = Field(..., description="Weighted combined score")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    model_loaded: bool = False
    model_name: str = ""
    device: str = ""
