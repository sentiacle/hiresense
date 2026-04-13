"""
HireSense AI - FastAPI Backend
Resume NER with BERT + BiLSTM + CRF
"""

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    CVAnalysisRequest, CVAnalysisResponse,
    EntityExtractionRequest, EntityExtractionResponse,
    TestGradeRequest, TestGradeResponse,
    PredictSuccessRequest, PredictSuccessResponse,
    HealthResponse, Entity, CategoryScore
)
from model_loader import get_model_manager
from scoring import (
    calculate_total_score,
    predict_success_probability,
    grade_test,
    group_entities_by_type
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    print("Starting HireSense AI Backend...")
    manager = get_model_manager()
    print(f"Model status: {manager.get_status()}")
    yield
    print("Shutting down HireSense AI Backend...")


app = FastAPI(
    title="HireSense AI API",
    description="Resume NER and Scoring API with BERT + BiLSTM + CRF",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    manager = get_model_manager()
    status = manager.get_status()
    
    return HealthResponse(
        status="ok",
        model_loaded=status["model_loaded"],
        model_name=status["model_name"],
        device=status["device"]
    )


@app.post("/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv(request: CVAnalysisRequest):
    """
    Analyze a CV using BERT + BiLSTM + CRF NER model
    
    - Extracts named entities (skills, experience, education, etc.)
    - Calculates weighted scores based on recruiter-defined weights
    - Returns detailed breakdown and total score
    """
    try:
        manager = get_model_manager()
        
        # Extract entities
        entities_raw, tokens, labels = manager.extract_entities(request.cv_text)
        
        # Convert to Entity objects
        entities = [
            Entity(
                text=e["text"],
                label=e["label"],
                start=e.get("start", 0),
                end=e.get("end", 0),
                confidence=e.get("confidence", 1.0)
            )
            for e in entities_raw
        ]
        
        # Calculate scores
        result = calculate_total_score(
            entities_raw,
            request.cv_text,
            request.required_skills,
            request.weights
        )
        
        # Convert scores to CategoryScore objects
        scores = {
            key: CategoryScore(
                score=value["score"],
                weight=value["weight"],
                entities=value["entities"],
                details=value["details"]
            )
            for key, value in result["scores"].items()
        }
        
        # Model info
        status = manager.get_status()
        model_info = {
            "model_name": status["model_name"],
            "device": status["device"],
            "model_type": "BERT + BiLSTM + CRF" if status["model_loaded"] else "Heuristic"
        }
        
        return CVAnalysisResponse(
            success=True,
            entities=entities,
            scores=scores,
            total_score=result["total_score"],
            breakdown=result["breakdown"],
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """
    Extract named entities from text (raw extraction without scoring)
    """
    try:
        manager = get_model_manager()
        entities_raw, tokens, labels = manager.extract_entities(request.text)
        
        entities = [
            Entity(
                text=e["text"],
                label=e["label"],
                start=e.get("start", 0),
                end=e.get("end", 0),
                confidence=e.get("confidence", 1.0)
            )
            for e in entities_raw
        ]
        
        return EntityExtractionResponse(
            success=True,
            entities=entities,
            tokens=tokens[:100],  # Limit tokens returned
            labels=labels[:100]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade-test", response_model=TestGradeResponse)
async def grade_test_endpoint(request: TestGradeRequest):
    """
    Grade test answers
    """
    try:
        result = grade_test(request.answers, request.correct_answers)
        
        return TestGradeResponse(
            success=True,
            score=result["score"],
            correct=result["correct"],
            total=result["total"],
            details=result["details"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-success", response_model=PredictSuccessResponse)
async def predict_success(request: PredictSuccessRequest):
    """
    Predict hiring success probability based on CV and test scores
    """
    try:
        probability = predict_success_probability(
            request.cv_score,
            request.test_score,
            request.cv_weight
        )
        
        combined_score = (
            request.cv_weight * request.cv_score +
            (1 - request.cv_weight) * request.test_score
        )
        
        passed = (
            request.cv_score >= request.min_cv_score and
            request.test_score >= request.min_test_score
        )
        
        return PredictSuccessResponse(
            success=True,
            probability=probability,
            passed=passed,
            combined_score=round(combined_score, 1)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For Vercel serverless deployment
def handler(event, context):
    """AWS Lambda / Vercel Serverless handler"""
    from mangum import Mangum
    asgi_handler = Mangum(app)
    return asgi_handler(event, context)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
