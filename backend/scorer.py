"""
HireSense AI — CV-vs-JD Scoring Helper
Integrates with your FastAPI scoring endpoint.

This is a thin wrapper. The actual logic lives in scoring.py.
Use score_cv() for the entity-based heuristic scorer (fast, no model needed).

Bug fix: weight keys are now normalised to plural form before being passed
to calculate_total_score(), so calling with either 'skill' or 'skills' always works.
"""

from scoring import calculate_total_score

# Canonical weight key set (plural)
_CANONICAL = {
    "skill":       "skills",
    "skills":      "skills",
    "experience":  "experience",
    "education":   "education",
    "project":     "projects",
    "projects":    "projects",
    "achievement": "achievements",
    "achievements":"achievements",
}


def score_cv(
    cv_entities: list,
    cv_text: str,
    required_skills: list,
    weights: dict = None,
) -> dict:
    """
    Score a CV against a job description.

    Args:
        cv_entities:     List of entity dicts [{label, text, start, end}]
                         from the NER model's extract() call.
        cv_text:         Raw CV text string (used as fallback for skill matching
                         when NER misses a skill that's clearly in the text).
        required_skills: List of skill strings extracted from the recruiter's JD.
                         Pass [] if no JD was provided; scoring falls back to
                         counting raw skill entities from CV text.
                         WARNING: passing [] means skills are NOT scored vs JD —
                         pass jd_text to CVAnalysisRequest to auto-extract skills.
        weights:         Category weights dict. Both plural ('skills', 'projects',
                         'achievements') and singular ('skill', 'project',
                         'achievement') keys are accepted. Values do NOT need
                         to sum to 100 — they are normalised internally.

    Returns:
        {
            "scores": {
                "skills":       {"score": float, "weight": float, "entities": [...], "details": str},
                "experience":   {...},
                "education":    {...},
                "projects":     {...},
                "achievements": {...},
            },
            "total_score": float,   # 0–100, weighted and normalised
            "jd_used":     bool,    # True if required_skills were provided/extracted
            "breakdown": {
                "skills":         [...],
                "experience":     [...],
                "education":      [...],
                "projects":       [...],
                "achievements":   [...],
                "certifications": [...],
                "organizations":  [...],
                "dates":          [...],
            }
        }
    """
    import logging
    if not required_skills:
        logging.warning(
            "[HireSense] score_cv called with empty required_skills. "
            "Skills will be scored from CV text only — NOT matched against a JD. "
            "To get JD-relative scores (typically 20-30 pts higher), ensure jd_text "
            "is passed in CVAnalysisRequest so required_skills are auto-extracted BEFORE "
            "this function is called. Check that your API handler forwards jd_text."
        )
    if weights is None:
        weights = {
            "skills": 40, "experience": 25, "projects": 15,
            "achievements": 10, "education": 10,
        }

    # Normalise keys to canonical plural form
    canonical_weights = {}
    for k, v in weights.items():
        canon = _CANONICAL.get(k.lower())
        if canon:
            canonical_weights[canon] = v

    # Fill any missing categories with defaults
    defaults = {"skills": 40, "experience": 25, "projects": 15,
                "achievements": 10, "education": 10}
    for cat, default_val in defaults.items():
        canonical_weights.setdefault(cat, default_val)

    result = calculate_total_score(
        entities=cv_entities,
        cv_text=cv_text,
        required_skills=required_skills,
        weights=canonical_weights,
    )
    # Annotate whether JD skill matching was active
    result["jd_used"] = bool(required_skills)
    result["skills_from_jd"] = required_skills
    return result
