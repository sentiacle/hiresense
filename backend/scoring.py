"""
HireSense AI - Entity-based Scoring Engine
Calculates CV scores based on extracted entities
"""

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict


def group_entities_by_type(entities: List[Dict]) -> Dict[str, List[str]]:
    """Group entities by their type"""
    grouped = defaultdict(list)
    
    for entity in entities:
        label = entity.get("label", "")
        text = entity.get("text", "")
        if text and label:
            grouped[label].append(text)
    
    return dict(grouped)


def calculate_skill_score(
    skill_entities: List[str],
    required_skills: List[str],
    cv_text: str
) -> Tuple[float, List[str], str]:
    """
    Calculate skills score based on extracted entities
    
    Returns: (score, matched_skills, details)
    """
    extracted_skills = set(s.lower() for s in skill_entities)
    required_lower = set(s.lower() for s in required_skills)
    cv_lower = cv_text.lower()
    
    # Match against required skills
    matched_required = extracted_skills & required_lower
    
    # Also check if skills appear in CV text (backup)
    for skill in required_lower:
        if skill in cv_lower and skill not in matched_required:
            matched_required.add(skill)
    
    # Score calculation
    if required_skills:
        required_match_rate = len(matched_required) / len(required_skills)
        required_score = required_match_rate * 70  # 70% weight for required skills
    else:
        required_score = 35  # Default if no required skills specified
    
    # Bonus for additional skills
    additional_skills = len(extracted_skills) - len(matched_required)
    additional_score = min(additional_skills * 3, 30)  # Up to 30 points
    
    total_score = min(required_score + additional_score, 100)
    
    # Details
    if required_skills:
        details = f"Matched {len(matched_required)}/{len(required_skills)} required skills. "
        details += f"Found {len(extracted_skills)} total skills."
    else:
        details = f"Found {len(extracted_skills)} skills."
    
    return total_score, list(extracted_skills), details


def calculate_experience_score(
    exp_entities: List[str],
    cv_text: str
) -> Tuple[float, List[str], str]:
    """
    Calculate experience score based on extracted entities
    
    Returns: (score, experience_items, details)
    """
    score = 0
    details_parts = []
    
    # Extract years from entities and text
    years = 0
    for exp in exp_entities:
        year_match = re.search(r'(\d+)\+?\s*years?', exp.lower())
        if year_match:
            years = max(years, int(year_match.group(1)))
    
    # Also check raw text
    year_matches = re.findall(r'(\d+)\+?\s*years?\s*(of\s+)?experience', cv_text.lower())
    for match in year_matches:
        years = max(years, int(match[0]))
    
    # Years-based score (up to 50 points)
    years_score = min(years * 10, 50)
    score += years_score
    details_parts.append(f"{years} years of experience")
    
    # Seniority level detection
    seniority_patterns = {
        "senior": 20,
        "lead": 25,
        "principal": 30,
        "staff": 25,
        "manager": 20,
        "director": 30
    }
    
    cv_lower = cv_text.lower()
    for level, points in seniority_patterns.items():
        if level in cv_lower:
            score += points
            details_parts.append(f"{level.title()} level position")
            break
    
    # Cap at 100
    score = min(score, 100)
    
    return score, exp_entities, ", ".join(details_parts) if details_parts else "No experience details found"


def calculate_education_score(
    edu_entities: List[str],
    cv_text: str
) -> Tuple[float, List[str], str]:
    """
    Calculate education score based on extracted entities
    
    Returns: (score, education_items, details)
    """
    score = 0
    details_parts = []
    cv_lower = cv_text.lower()
    
    # Degree levels
    degree_scores = {
        "phd": 40,
        "doctorate": 40,
        "master": 30,
        "m.s.": 30,
        "m.s": 30,
        "mba": 30,
        "bachelor": 20,
        "b.s.": 20,
        "b.s": 20,
        "b.sc": 20,
    }
    
    # Check entities first
    found_degree = None
    for edu in edu_entities:
        edu_lower = edu.lower()
        for degree, points in degree_scores.items():
            if degree in edu_lower:
                if points > score:
                    score = points
                    found_degree = degree.upper()
                break
    
    # Also check raw text
    for degree, points in degree_scores.items():
        if degree in cv_lower:
            if points > score:
                score = points
                found_degree = degree.upper()
    
    if found_degree:
        details_parts.append(f"{found_degree} degree")
    
    # GPA bonus
    gpa_match = re.search(r'gpa[\s:]+([0-9.]+)', cv_lower)
    if gpa_match:
        gpa = float(gpa_match.group(1))
        if gpa >= 3.8:
            score += 20
            details_parts.append(f"GPA: {gpa} (excellent)")
        elif gpa >= 3.5:
            score += 15
            details_parts.append(f"GPA: {gpa} (very good)")
        elif gpa >= 3.0:
            score += 10
            details_parts.append(f"GPA: {gpa} (good)")
    
    # University prestige (simple keyword matching)
    prestige_unis = ["mit", "stanford", "harvard", "berkeley", "caltech", "cmu", "princeton"]
    for uni in prestige_unis:
        if uni in cv_lower:
            score += 15
            details_parts.append(f"Top university")
            break
    
    # Relevant field bonus
    relevant_fields = ["computer science", "data science", "engineering", "mathematics", "statistics"]
    for field in relevant_fields:
        if field in cv_lower:
            score += 10
            details_parts.append(f"Relevant field: {field.title()}")
            break
    
    score = min(score, 100)
    
    return score, edu_entities, ", ".join(details_parts) if details_parts else "No education details found"


def calculate_projects_score(
    proj_entities: List[str],
    cv_text: str
) -> Tuple[float, List[str], str]:
    """
    Calculate projects score based on extracted entities
    
    Returns: (score, projects, details)
    """
    score = 0
    projects = list(proj_entities)
    cv_lower = cv_text.lower()
    
    # Project-related keywords
    project_keywords = [
        "built", "developed", "created", "designed", "implemented",
        "architected", "led", "launched", "deployed", "maintained"
    ]
    
    keyword_count = sum(1 for kw in project_keywords if kw in cv_lower)
    score += min(keyword_count * 8, 40)
    
    # Portfolio/GitHub presence
    if "github" in cv_lower or "portfolio" in cv_lower:
        score += 20
    
    # Open source contribution
    if "open source" in cv_lower or "opensource" in cv_lower:
        score += 15
    
    # Hackathon/competition
    if "hackathon" in cv_lower or "competition" in cv_lower:
        score += 10
    
    # Number of projects mentioned
    project_count = len(projects) + keyword_count // 2
    score += min(project_count * 5, 25)
    
    score = min(score, 100)
    
    details = f"Found {len(projects)} project entities, {keyword_count} action verbs"
    
    return score, projects, details


def calculate_achievements_score(
    ach_entities: List[str],
    cert_entities: List[str],
    cv_text: str
) -> Tuple[float, List[str], str]:
    """
    Calculate achievements/certifications score
    
    Returns: (score, achievements, details)
    """
    score = 0
    achievements = list(ach_entities) + list(cert_entities)
    cv_lower = cv_text.lower()
    details_parts = []
    
    # Certifications (20 points each, max 40)
    cert_count = len(cert_entities)
    cert_keywords = ["certified", "certification", "certificate"]
    for kw in cert_keywords:
        cert_count += cv_lower.count(kw)
    cert_score = min(cert_count * 20, 40)
    score += cert_score
    if cert_count > 0:
        details_parts.append(f"{cert_count} certifications")
    
    # Awards/honors (15 points each, max 30)
    award_keywords = ["award", "prize", "winner", "honor", "recognition"]
    award_count = sum(cv_lower.count(kw) for kw in award_keywords)
    award_score = min(award_count * 15, 30)
    score += award_score
    if award_count > 0:
        details_parts.append(f"{award_count} awards/honors")
    
    # Publications/patents (20 points each, max 30)
    pub_keywords = ["published", "publication", "patent", "paper", "journal"]
    pub_count = sum(cv_lower.count(kw) for kw in pub_keywords)
    pub_score = min(pub_count * 20, 30)
    score += pub_score
    if pub_count > 0:
        details_parts.append(f"{pub_count} publications/patents")
    
    # Quantifiable achievements (percentage improvements)
    percent_matches = re.findall(r'\d+%\s*(increase|improvement|reduction|growth)', cv_lower)
    if percent_matches:
        score += min(len(percent_matches) * 10, 20)
        details_parts.append(f"{len(percent_matches)} quantified achievements")
    
    score = min(score, 100)
    
    return score, achievements, ", ".join(details_parts) if details_parts else "No achievements found"


def calculate_total_score(
    entities: List[Dict],
    cv_text: str,
    required_skills: List[str],
    weights: Dict[str, float]
) -> Dict:
    """
    Calculate total CV score with breakdown
    
    Args:
        entities: Extracted NER entities
        cv_text: Raw CV text
        required_skills: Skills required for the job
        weights: Category weights (must sum to 100)
    
    Returns:
        Dictionary with scores, breakdown, and total
    """
    # Group entities
    grouped = group_entities_by_type(entities)
    
    # Calculate individual scores
    skill_score, skill_items, skill_details = calculate_skill_score(
        grouped.get("SKILL", []),
        required_skills,
        cv_text
    )
    
    exp_score, exp_items, exp_details = calculate_experience_score(
        grouped.get("EXP", []),
        cv_text
    )
    
    edu_score, edu_items, edu_details = calculate_education_score(
        grouped.get("EDU", []),
        cv_text
    )
    
    proj_score, proj_items, proj_details = calculate_projects_score(
        grouped.get("PROJ", []),
        cv_text
    )
    
    ach_score, ach_items, ach_details = calculate_achievements_score(
        grouped.get("ACH", []),
        grouped.get("CERT", []),
        cv_text
    )
    
    # Build scores dictionary
    scores = {
        "skills": {
            "score": skill_score,
            "weight": weights.get("skills", 40),
            "entities": skill_items,
            "details": skill_details
        },
        "experience": {
            "score": exp_score,
            "weight": weights.get("experience", 25),
            "entities": exp_items,
            "details": exp_details
        },
        "education": {
            "score": edu_score,
            "weight": weights.get("education", 10),
            "entities": edu_items,
            "details": edu_details
        },
        "projects": {
            "score": proj_score,
            "weight": weights.get("projects", 15),
            "entities": proj_items,
            "details": proj_details
        },
        "achievements": {
            "score": ach_score,
            "weight": weights.get("achievements", 10),
            "entities": ach_items,
            "details": ach_details
        }
    }
    
    # Calculate weighted total
    total_weight = sum(s["weight"] for s in scores.values())
    if total_weight > 0:
        total_score = sum(
            s["score"] * s["weight"] / total_weight
            for s in scores.values()
        )
    else:
        total_score = 0
    
    # Build breakdown
    breakdown = {
        "skills": skill_items,
        "experience": exp_items,
        "education": edu_items,
        "projects": proj_items,
        "achievements": ach_items,
        "organizations": grouped.get("ORG", []),
        "dates": grouped.get("DATE", [])
    }
    
    return {
        "scores": scores,
        "total_score": round(total_score, 1),
        "breakdown": breakdown
    }


def predict_success_probability(
    cv_score: float,
    test_score: float,
    cv_weight: float = 0.6
) -> float:
    """
    Predict hiring success probability using sigmoid
    
    Args:
        cv_score: CV score (0-100)
        test_score: Test score (0-100)
        cv_weight: Weight for CV score (default 0.6)
    
    Returns:
        Probability percentage (0-100)
    """
    test_weight = 1 - cv_weight
    
    # Normalize to 0-1
    cv_norm = cv_score / 100
    test_norm = test_score / 100
    
    # Combined score
    combined = cv_weight * cv_norm + test_weight * test_norm
    
    # Sigmoid with steepness 10, centered at 0.5
    probability = 1 / (1 + math.exp(-10 * (combined - 0.5)))
    
    return round(probability * 100, 1)


def grade_test(
    answers: List[int],
    correct_answers: List[int]
) -> Dict:
    """
    Grade test answers
    
    Returns:
        Dictionary with score, correct count, total, and details
    """
    if len(answers) != len(correct_answers):
        raise ValueError("Answer count mismatch")
    
    total = len(correct_answers)
    correct = sum(1 for a, c in zip(answers, correct_answers) if a == c)
    score = (correct / total * 100) if total > 0 else 0
    
    details = [
        {
            "question": i + 1,
            "user_answer": answers[i],
            "correct_answer": correct_answers[i],
            "is_correct": answers[i] == correct_answers[i]
        }
        for i in range(total)
    ]
    
    return {
        "score": round(score, 1),
        "correct": correct,
        "total": total,
        "details": details
    }
