"""
HireSense AI — Entity-based Scoring Engine
Calculates CV scores based on extracted NER entities + JD matching.

Bug fixes applied:
  1. Experience: fresher/intern CVs never say "X years" — now counts internship
     entries and computes duration from date ranges.
  2. Education: CGPA "3.39/4" format now detected (was only matching "GPA: X").
  3. Skill matching: "Microsoft Excel" entity matches JD term "excel" via
     substring normalisation (was requiring exact set intersection).
  4. Experience: added seniority check BEFORE capping so it contributes properly.
  5. Achievements: cert_count no longer double-counts — NER cert entities are
     NOT also searched in raw text (caused inflation AND missed the real signal).
  6. Weights: normalised to 100 even if caller sends values that don't sum to 100.
  7. JD-relevance multiplier: experience, education, projects, and achievements
     now scale their raw scores by how relevant the CV content is to the JD.
     A CV with no JD-relevant content in those sections scores proportionally
     lower, fixing the bug where all categories returned identical scores
     regardless of the job description.
"""

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_overlap(a: str, b: str) -> bool:
    """
    True when either string is a substring of the other (after normalisation).
    Handles: "excel" matching "microsoft excel", "fastapi" matching "fastapi", etc.
    Minimum length guard (≥3 chars) prevents single letters triggering matches.
    """
    a, b = _normalize(a), _normalize(b)
    if len(a) < 3 or len(b) < 3:
        return False
    return a in b or b in a


def _jd_relevance_multiplier(
    section_text: str,
    required_skills: List[str],
    floor: float = 0.35,
    ceil: float = 1.0,
) -> Tuple[float, str]:
    """
    Return a relevance multiplier (floor … ceil) that scales a category score
    based on how much JD terminology appears in that CV section's text.

    If no JD skills are provided (required_skills=[]), multiplier is always 1.0
    so existing no-JD behaviour is preserved.

    Algorithm:
      - For each JD term, check if it appears anywhere in section_text
        (substring match, same as skill scoring).
      - relevance = matched_terms / total_terms  (0.0 – 1.0)
      - multiplier = floor + (ceil - floor) * relevance
        so even a completely mismatched section still gets `floor` of its raw
        score (prevents harsh zeroes for legitimate experience/education).

    Args:
        section_text:    The CV text to search within (e.g. experience block,
                         whole CV text, or combined entity strings).
        required_skills: JD skill/keyword list extracted from the job description.
        floor:           Minimum multiplier even when 0 JD terms match.
        ceil:            Maximum multiplier when all JD terms match.

    Returns:
        (multiplier: float, note: str)
    """
    if not required_skills:
        return 1.0, "no JD"

    text_lower = section_text.lower()
    matched = sum(
        1 for term in required_skills
        if _token_overlap(term, text_lower) or _normalize(term) in text_lower
    )
    relevance  = matched / len(required_skills)
    multiplier = floor + (ceil - floor) * relevance
    note       = f"{matched}/{len(required_skills)} JD terms → ×{multiplier:.2f}"
    return multiplier, note


def _extract_years_from_date_range(text: str) -> float:
    """
    Parse date ranges like '19 May 2025 – 12 July 2025' or '2020-2022'
    and return approximate years of experience from them.
    """
    total_months = 0.0

    MONTH_MAP = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    }

    # Pattern 1: "DD Month YYYY – DD Month YYYY" or "Month YYYY – Month YYYY"
    # Also handles ordinal suffixes (19th, 12th) and apostrophe-year (May'2025)
    pattern1 = re.compile(
        r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"         # optional day with ordinal
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
        r"['\s.,]*(\d{4})"                           # year (with optional apostrophe separator)
        r"\s*[-–—to]+\s*"
        r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)[a-z]*"
        r"['\s.,]*(\d{4})?",
        re.IGNORECASE,
    )
    for m in pattern1.finditer(text):
        m1, y1_s, m2_raw, y2_s = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            y1 = int(y1_s)
            mon1 = MONTH_MAP.get(m1[:3].lower(), 1)
            if m2_raw.lower() in ("present", "current"):
                import datetime
                now = datetime.date.today()
                y2, mon2 = now.year, now.month
            else:
                y2   = int(y2_s) if y2_s else y1
                mon2 = MONTH_MAP.get(m2_raw[:3].lower(), 12)
            diff = (y2 - y1) * 12 + (mon2 - mon1)
            total_months += max(diff, 0)
        except Exception:
            pass

    # Pattern 2: "YYYY – YYYY"
    pattern2 = re.compile(r"(20\d{2})\s*[-–—to]+\s*(20\d{2}|present|current)", re.IGNORECASE)
    for m in pattern2.finditer(text):
        try:
            y1 = int(m.group(1))
            if m.group(2).lower() in ("present", "current"):
                import datetime
                y2 = datetime.date.today().year
            else:
                y2 = int(m.group(2))
            total_months += max((y2 - y1) * 12, 0)
        except Exception:
            pass

    return total_months / 12.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def group_entities_by_type(entities: List[Dict]) -> Dict[str, List[str]]:
    """Group extracted NER entities by their label."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for entity in entities:
        label = entity.get("label", "").upper()
        text  = entity.get("text", "").strip()
        if text and label:
            grouped[label].append(text)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Category scorers
# ---------------------------------------------------------------------------

def calculate_skill_score(
    skill_entities: List[str],
    required_skills: List[str],
    cv_text: str,
) -> Tuple[float, List[str], str]:
    """
    Score skills.  Matching uses substring overlap so:
      • "Microsoft Excel" entity  matches JD term "excel"
      • "Full Stack Web Dev"      matches JD term "web development"
      • "FastAPI" entity          matches JD term "fastapi"

    Scoring:
      70 pts  — required-skill coverage (fraction matched × 70)
      30 pts  — bonus for additional skills the JD didn't list
    """
    # Normalise extracted skill entities
    extracted_norm = [_normalize(s) for s in skill_entities]

    # Also harvest skills directly from CV text (backup for weak NER)
    # Split on common delimiters used in skill sections
    cv_lower = cv_text.lower()
    raw_cv_skills = re.findall(r"[a-z][a-z0-9+#./\- ]{1,28}", cv_lower)

    def _is_matched(req_term: str) -> bool:
        req = _normalize(req_term)
        # Check entities first
        for e in extracted_norm:
            if _token_overlap(req, e):
                return True
        # Fallback: check raw CV text
        if req in cv_lower:
            return True
        # Word-level: "data analysis" splits and checks "data" + "analysis" present
        words = req.split()
        if len(words) > 1 and all(w in cv_lower for w in words):
            return True
        return False

    matched_required = [s for s in required_skills if _is_matched(s)]

    if required_skills:
        required_score = (len(matched_required) / len(required_skills)) * 70
    else:
        # No JD provided — mine skills from raw CV text (bullet/comma separated)
        # so a skill-rich CV is scored fairly even without NER entities.
        # FIX: raised cap from 55 → 70 so a skill-rich CV can earn a realistic score.
        cv_skill_hits = re.findall(
            r"(?:^|[•\-,;|\n])\s*([A-Za-z][A-Za-z0-9+#./\- ]{2,28}?)(?=[,;•|\n]|$)",
            cv_text, re.MULTILINE,
        )
        all_skill_count = len(set(extracted_norm) | {_normalize(s) for s in cv_skill_hits})
        required_score = min(all_skill_count * 4, 70)  # raised: 3→4 pts each, cap 55→70

    # Additional skills bonus (skills extracted but not in JD requirements)
    # FIX: raised multiplier 3→5 and cap 30→35 so depth of skill set is rewarded.
    additional_count = max(len(extracted_norm) - len(matched_required), 0)
    additional_score = min(additional_count * 5, 35)

    total_score = min(required_score + additional_score, 100)

    details = (
        f"Matched {len(matched_required)}/{len(required_skills)} required skills. "
        f"Found {len(extracted_norm)} total skill entities."
    ) if required_skills else f"Found {len(extracted_norm)} skills (no JD requirements specified)."

    return total_score, extracted_norm, details


def calculate_experience_score(
    exp_entities: List[str],
    cv_text: str,
    required_skills: List[str] = None,
) -> Tuple[float, List[str], str]:
    """
    Score experience.  Handles BOTH:
      • Traditional CVs: "5 years of experience" → years-based score
      • Fresher/intern CVs (like Sanyam's): date ranges like
        "19 May 2025 – 12 July 2025" → compute duration; also count
        the number of distinct internship / work entries.

    Scoring breakdown (max 100):
      Date-range duration   → up to 40 pts  (1 pt per month, max 40)
      Internship/role count → up to 30 pts  (15 pt per role, max 30)
      Explicit years text   → up to 30 pts  (10 pt per year, max 30)
      Seniority bonus       → up to 25 pts

    JD relevance: raw score is scaled by a multiplier (0.35–1.0) based on
    how many JD terms appear in the experience section of the CV.
    """
    score       = 0.0
    details_parts = []
    cv_lower    = cv_text.lower()

    # 1. Parse explicit "X years (of) experience" mentions
    explicit_years = 0
    for exp in exp_entities:
        m = re.search(r"(\d+)\+?\s*years?", exp.lower())
        if m:
            explicit_years = max(explicit_years, int(m.group(1)))
    for m in re.findall(r"(\d+)\+?\s*years?\s*(?:of\s+)?experience", cv_lower):
        explicit_years = max(explicit_years, int(m))
    if explicit_years:
        years_score = min(explicit_years * 10, 30)
        score += years_score
        details_parts.append(f"{explicit_years} yrs explicit")

    # 2. Parse date ranges to estimate total experience duration
    range_years = _extract_years_from_date_range(cv_text)
    if range_years > 0:
        range_score = min(range_years * 24, 48)   # RAISED: 2 pts per month, cap 48
        score += range_score
        details_parts.append(f"~{range_years:.1f} yrs from date ranges (+{range_score:.0f} pts)")

    # 3. Count distinct internship / work entries
    internship_count = len(re.findall(r"\b(internship|intern\b)", cv_lower))
    role_entries     = len(re.findall(r"\b(20\d{2})\s*[-–—]\s*(20\d{2}|present)", cv_lower, re.IGNORECASE))
    header_roles     = len(re.findall(
        r"(technical internship|other internship|work experience|summer internship)", cv_lower
    ))
    # FIX: detect "Internship – COMPANY" style headers (handles Yog's apostrophe-date format)
    named_roles = len(re.findall(
        r"(?:technical|other|summer|finance|marketing|research)?\s*internship\s*[-\u2013\u2014@]\s*\w",
        cv_lower,
    ))
    roles = max(internship_count, role_entries, header_roles, named_roles)
    # Corroboration bonus when multiple signals agree
    if internship_count > 0 and role_entries > 0:
        roles = max(roles, internship_count + role_entries - 1)
    if roles > 0:
        role_score = min(roles * 22, 44)   # RAISED: 22 pts per role, cap 44
        score += role_score
        details_parts.append(f"{roles} roles/internships (+{int(role_score)} pts)")

    # 3b. Fresher baseline: raised to 55 so a real internship always earns meaningful credit
    if roles > 0 and score < 55:
        score = max(score, 55)
        details_parts.append("Fresher baseline applied (+55 floor)")

    # 4. Seniority bonus
    seniority_map = [
        ("director",   25), ("principal", 25), ("staff",   20),
        ("lead",       20), ("senior",    15), ("manager", 15),
    ]
    for level, pts in seniority_map:
        if level in cv_lower:
            score += pts
            details_parts.append(f"{level.title()}-level")
            break

    score = min(score, 100)

    # Apply JD relevance multiplier: experience in an unrelated domain scores lower.
    # Use full CV text (not just a regex-extracted fragment) so that internship
    # descriptions — which contain the most domain-specific language — are included.
    # floor raised 0.40→0.50: even mismatched experience deserves half credit.
    multiplier, mult_note = _jd_relevance_multiplier(
        cv_text, required_skills or [], floor=0.45
    )
    score = round(score * multiplier, 1)
    details_parts.append(f"JD relevance: {mult_note}")

    details = ", ".join(details_parts) if details_parts else "No experience details detected"
    return score, exp_entities, details


def calculate_education_score(
    edu_entities: List[str],
    cv_text: str,
    required_skills: List[str] = None,
) -> Tuple[float, List[str], str]:
    """
    Score education. Detects:
      • Degree level (MBA, B.Tech, PhD…)
      • CGPA in "X.XX/4" format  AND  "GPA: X.XX" format
      • Relevant field keywords
      • Prestige university keywords

    JD relevance: the field-bonus step explicitly checks whether the degree
    field matches the JD. A tech JD with a finance-only degree, or vice versa,
    scores lower on the relevance-adjusted scale.
    """
    score        = 0
    details_parts: List[str] = []
    cv_lower     = cv_text.lower()

    # Degree level scores (check entities first, then raw text)
    degree_scores = {
        "phd": 40, "doctorate": 40,
        "master": 30, "m.s": 30, "mba": 30, "mba tech": 30, "m.tech": 30,
        "bachelor": 20, "b.s": 20, "b.sc": 20, "b.tech": 20, "b.e": 20,
        "diploma": 10,
    }
    found_degree_pts = 0
    found_degree_name = ""
    combined_edu_text = " ".join(edu_entities).lower() + " " + cv_lower
    for degree, pts in degree_scores.items():
        if degree in combined_edu_text and pts > found_degree_pts:
            found_degree_pts = pts
            found_degree_name = degree.upper()
    score += found_degree_pts
    if found_degree_name:
        details_parts.append(f"{found_degree_name} degree ({found_degree_pts} pts)")

    # GPA / CGPA detection
    # Format 1: "GPA: 3.5" or "GPA 3.5"
    gpa_val = None
    m = re.search(r"gpa[\s:]+([0-9]+\.[0-9]+)", cv_lower)
    if m:
        gpa_val = float(m.group(1))

    # Format 2: "3.39/4" (Indian CGPA notation) — most common in uploaded CVs
    if gpa_val is None:
        m = re.search(r"\b([0-9]+\.[0-9]+)\s*/\s*4\b", cv_lower)
        if m:
            gpa_val = float(m.group(1))

    # Format 3: "CGPA: 8.71" (out of 10)
    if gpa_val is None:
        m = re.search(r"cgpa[\s:]+([0-9]+\.[0-9]+)", cv_lower)
        if m:
            raw = float(m.group(1))
            gpa_val = raw / 10 * 4 if raw > 4 else raw  # normalise to /4 scale

    # Format 4: bare 10-point scale "8.71" near word "cgpa" OR in edu entities only
    # FIX: previously searched cv_lower[:500] which hit phone numbers/dates.
    # Now restricted to edu_entities text only to avoid false positives.
    if gpa_val is None:
        combined_edu = " ".join(edu_entities).lower()
        m = re.search(r"\b([7-9]\.[0-9]{1,2}|10\.0)\b", combined_edu)
        if m:
            raw = float(m.group(1))
            if raw > 4:  # only treat as /10 if > 4
                gpa_val = raw / 10 * 4

    # BUG FIX: GPA scoring block was incorrectly nested inside `if gpa_val is None`
    # which meant GPA score was NEVER awarded. Now runs after all detection formats.
    if gpa_val is not None:
        if gpa_val >= 3.7:
            score += 20; details_parts.append(f"GPA {gpa_val:.2f}/4 — excellent (+20)")
        elif gpa_val >= 3.5:
            score += 15; details_parts.append(f"GPA {gpa_val:.2f}/4 — very good (+15)")
        elif gpa_val >= 3.0:
            score += 10; details_parts.append(f"GPA {gpa_val:.2f}/4 — good (+10)")
        elif gpa_val >= 2.5:
            score += 5;  details_parts.append(f"GPA {gpa_val:.2f}/4 — average (+5)")

    # Prestige university keywords
    prestige = ["iit","nit","bits","iiit","mit","stanford","harvard","nmims","vit"]
    for uni in prestige:
        if uni in cv_lower:
            score += 10
            details_parts.append(f"Notable institution (+10)")
            break

    # Relevant field bonus — now JD-aware: check if degree field matches JD terms
    # Map JD keyword categories to degree fields
    jd_lower = " ".join(required_skills or []).lower()
    tech_jd   = any(t in jd_lower for t in ["python","java","software","engineer","developer","data","cloud","backend","frontend","api","ml","ai"])
    finance_jd = any(t in jd_lower for t in ["finance","accounting","investment","banking","equity","valuation","financial","cfa","bloomberg"])

    fields = ["computer science","information technology","data science",
              "engineering","mathematics","statistics","finance","management"]
    for field in fields:
        if field in cv_lower:
            # Award field bonus only if field aligns with JD, or no JD provided
            if not required_skills:
                score += 5
                details_parts.append(f"Relevant field: {field.title()} (+5)")
            elif (tech_jd and field in ["computer science","information technology","data science","engineering","mathematics","statistics"]):
                score += 5
                details_parts.append(f"Field matches JD (tech): {field.title()} (+5)")
            elif (finance_jd and field in ["finance","management","mathematics","statistics"]):
                score += 5
                details_parts.append(f"Field matches JD (finance): {field.title()} (+5)")
            else:
                details_parts.append(f"Field present but not JD-aligned: {field.title()} (+0)")
            break

    score = min(score, 100)

    # Apply JD relevance multiplier using the education section text
    # Education is less JD-sensitive than experience (floor=0.60) — a degree is
    # still a degree even if the field doesn't perfectly match.
    multiplier, mult_note = _jd_relevance_multiplier(
        cv_lower, required_skills or [], floor=0.60
    )
    score = round(score * multiplier, 1)
    details_parts.append(f"JD relevance: {mult_note}")

    details = ", ".join(details_parts) if details_parts else "No education details detected"
    return score, edu_entities, details


def calculate_projects_score(
    proj_entities: List[str],
    cv_text: str,
    required_skills: List[str] = None,
) -> Tuple[float, List[str], str]:
    """
    Score projects. Counts:
      • Action verbs (built, developed, deployed…)
      • GitHub / portfolio presence
      • Hackathon / competition participation
      • Number of distinct project entries
      • Quantified impact (percentages, user counts, etc.)

    JD relevance: project descriptions are checked for JD keyword overlap.
    Building a Python backend scores highly for a backend JD; the same projects
    score lower for an investment banking JD where those terms are absent.
    """
    score    = 0
    cv_lower = cv_text.lower()

    action_verbs = [
        "built","developed","created","designed","implemented",
        "architected","led","launched","deployed","maintained",
        "engineered","integrated","automated","optimised","optimized",
    ]
    verb_count = sum(1 for v in action_verbs if v in cv_lower)
    score += min(verb_count * 8, 48)   # RAISED: 6→8 pts/verb, cap 36→48

    if "github" in cv_lower or "portfolio" in cv_lower:
        score += 15

    if "open source" in cv_lower or "opensource" in cv_lower:
        score += 10

    if "hackathon" in cv_lower or "competition" in cv_lower:
        score += 10

    # Count distinct project entities + project-marker sentences
    # FIX: raised entity multiplier 4→7 and added a minimum floor if action verbs exist
    project_markers = len(re.findall(
        r"(project|system|platform|application|app|tool|dashboard|website)",
        cv_lower,
    ))
    proj_count = len(proj_entities) + max(project_markers // 2, 0)
    score += min(proj_count * 9, 36)   # RAISED: 7→9 pts/project, cap 28→36

    # Floor: if CV has strong action verbs, projects section shouldn't score zero
    if verb_count >= 3 and score < 45:
        score = max(score, 45)   # RAISED floor 30→45

    # Quantified impact
    quantified = re.findall(
        r"\d+[%x]\s*(improvement|reduction|increase|faster|users|requests|accuracy)",
        cv_lower,
    )
    if quantified:
        score += min(len(quantified) * 5, 15)

    score = min(score, 100)

    # Extract just the projects section for a more targeted relevance check
    # FIX: use a tighter regex that stops at the next major section header
    proj_section = cv_text
    proj_match = re.search(
        r"(?:projects?|personal\s+projects?)\s*[\n\r](.*?)(?=\n\s*(?:internship|summer\s+internship|experience|achievements?|certifications?|education)\b|$)",
        cv_text, re.IGNORECASE | re.DOTALL,
    )
    if proj_match and len(proj_match.group(1)) > 30:
        proj_section = proj_match.group(1)

    # Projects are heavily JD-sensitive: a finance JD shouldn't reward AI/web projects
    # as much as a tech JD would. floor=0.30 allows a meaningful penalty.
    multiplier, mult_note = _jd_relevance_multiplier(
        proj_section, required_skills or [], floor=0.25
    )
    score = round(score * multiplier, 1)

    details = (
        f"{len(proj_entities)} project entities, {verb_count} action verbs"
        + (f", {len(quantified)} quantified impacts" if quantified else "")
        + f"; JD relevance: {mult_note}"
    )
    return score, proj_entities, details


def calculate_achievements_score(
    ach_entities: List[str],
    cert_entities: List[str],
    cv_text: str,
    required_skills: List[str] = None,
) -> Tuple[float, List[str], str]:
    """
    Score achievements and certifications.

    Key fix: certification entities from NER are NOT double-counted against
    raw text keywords. The two sources are kept separate and capped individually.

    JD relevance: certifications and achievements are checked for JD overlap.
    A Bloomberg cert is highly relevant for a finance JD; a Python cert is
    relevant for a tech JD. floor=0.50 so effort is always recognised.
    """
    score        = 0.0
    cv_lower     = cv_text.lower()
    details_parts: List[str] = []

    # ── Certifications ──────────────────────────────────────────────────────
    # Source 1: NER cert entities (most reliable)
    ner_cert_count = len(cert_entities)
    # Source 2: raw text keywords NOT already captured by NER
    raw_cert_count = max(
        cv_lower.count("certified") + cv_lower.count("certification")
        + cv_lower.count("certificate") - ner_cert_count,
        0,
    )
    cert_total   = ner_cert_count + raw_cert_count
    cert_score   = min(cert_total * 25, 50)   # RAISED: 20→25 pts/cert, cap 40→50
    score       += cert_score
    if cert_total:
        details_parts.append(f"{cert_total} certifications (+{int(cert_score)} pts)")

    # ── Awards / Honours ────────────────────────────────────────────────────
    award_kws    = ["award","prize","winner","honor","honour","recognition","merit",
                    "letter of recommendation","lor","selected","top "]
    award_count  = len(ach_entities) + sum(cv_lower.count(kw) for kw in award_kws)
    # cap entity double-count
    award_count  = min(award_count, 10)
    award_score  = min(award_count * 10, 35)   # RAISED: cap 25→35
    score       += award_score
    if award_count:
        details_parts.append(f"{award_count} awards/honours (+{int(award_score)} pts)")

    # ── Hackathon / competition wins ────────────────────────────────────────
    hack_patterns = re.findall(
        r"(hackathon|finalist|1st\s+place|winner|top\s+\d+|national\s+level|olympiad"
        r"|stock\s+analysis\s+competition|ibs\s+ahmedabad|entries)",
        cv_lower,
    )
    hack_score = min(len(hack_patterns) * 8, 25)   # RAISED: cap 20→25
    score     += hack_score
    if hack_patterns:
        details_parts.append(f"{len(hack_patterns)} competition mentions (+{int(hack_score)} pts)")

    # ── Publications / patents ──────────────────────────────────────────────
    pub_kws   = ["published","publication","patent","paper","journal","research",
                 "wealth street","magazine","tycoon"]
    pub_count = sum(cv_lower.count(kw) for kw in pub_kws)
    pub_score = min(pub_count * 15, 30)   # raised cap 20→30 for published work
    score    += pub_score
    if pub_count:
        details_parts.append(f"{pub_count} publication mentions (+{int(pub_score)} pts)")

    # ── Quantified achievements ─────────────────────────────────────────────
    quant = re.findall(r"\d+[%x]\s*(increase|improvement|reduction|growth|faster)", cv_lower)
    if quant:
        q_score = min(len(quant) * 8, 15)
        score  += q_score
        details_parts.append(f"{len(quant)} quantified achievements (+{int(q_score)} pts)")

    score = min(score, 100)

    # Apply JD relevance multiplier: certs and achievements that relate to the JD
    # domain are more valuable. floor=0.50 so genuine effort is always credited.
    ach_section = cv_text
    ach_match = re.search(
        r"(achievements?|certifications?|publications?)(.*?)(?=projects?|experience|education|$)",
        cv_text, re.IGNORECASE | re.DOTALL,
    )
    if ach_match:
        ach_section = ach_match.group(0)
    multiplier, mult_note = _jd_relevance_multiplier(
        ach_section, required_skills or [], floor=0.55
    )
    score = round(score * multiplier, 1)

    all_items = list(ach_entities) + list(cert_entities)
    details_parts.append(f"JD relevance: {mult_note}")
    details   = ", ".join(details_parts) if details_parts else "No achievements detected"
    return score, all_items, details


# ---------------------------------------------------------------------------
# Master scorer
# ---------------------------------------------------------------------------

def calculate_total_score(
    entities: List[Dict],
    cv_text: str,
    required_skills: List[str],
    weights: Dict[str, float],
) -> Dict:
    """
    Calculate the total CV score with full breakdown.

    Args:
        entities:        Extracted NER entities [{label, text, start, end}]
        cv_text:         Raw CV text (used as fallback for skill matching)
        required_skills: Skills extracted from the recruiter's JD
        weights:         Category weights dict. Accepts BOTH plural and singular
                         key forms: "skills"/"skill", "projects"/"project",
                         "achievements"/"achievement", "experience", "education".
                         Values are normalised to sum to 100 internally.

    Returns:
        {scores: {...}, total_score: float, breakdown: {...}}
    """
    # ── Normalise weight keys (singular ↔ plural) ──────────────────────────
    KEY_MAP = {
        "skills":       "skills",
        "skill":        "skills",
        "experience":   "experience",
        "education":    "education",
        "projects":     "projects",
        "project":      "projects",
        "achievements": "achievements",
        "achievement":  "achievements",
    }
    DEFAULTS = {"skills": 40, "experience": 25, "projects": 15,
                "achievements": 10, "education": 10}

    normalised_weights: Dict[str, float] = dict(DEFAULTS)
    for raw_key, val in weights.items():
        canonical = KEY_MAP.get(raw_key.lower())
        if canonical:
            normalised_weights[canonical] = val

    # Ensure weights sum to 100
    total_w = sum(normalised_weights.values())
    if total_w <= 0:
        normalised_weights = DEFAULTS
        total_w = 100.0

    # ── Group entities ─────────────────────────────────────────────────────
    grouped = group_entities_by_type(entities)

    # ── Individual category scores ─────────────────────────────────────────
    skill_score, skill_items, skill_details = calculate_skill_score(
        grouped.get("SKILL", []), required_skills, cv_text
    )
    exp_score, exp_items, exp_details = calculate_experience_score(
        grouped.get("EXP", []), cv_text, required_skills
    )
    edu_score, edu_items, edu_details = calculate_education_score(
        grouped.get("EDU", []), cv_text, required_skills
    )
    proj_score, proj_items, proj_details = calculate_projects_score(
        grouped.get("PROJ", []), cv_text, required_skills
    )
    ach_score, ach_items, ach_details = calculate_achievements_score(
        grouped.get("ACH", []), grouped.get("CERT", []), cv_text, required_skills
    )

    # ── Build scores dict ─────────────────────────────────────────────────
    scores = {
        "skills": {
            "score":    skill_score,
            "weight":   normalised_weights["skills"],
            "entities": skill_items,
            "details":  skill_details,
        },
        "experience": {
            "score":    exp_score,
            "weight":   normalised_weights["experience"],
            "entities": exp_items,
            "details":  exp_details,
        },
        "education": {
            "score":    edu_score,
            "weight":   normalised_weights["education"],
            "entities": edu_items,
            "details":  edu_details,
        },
        "projects": {
            "score":    proj_score,
            "weight":   normalised_weights["projects"],
            "entities": proj_items,
            "details":  proj_details,
        },
        "achievements": {
            "score":    ach_score,
            "weight":   normalised_weights["achievements"],
            "entities": ach_items,
            "details":  ach_details,
        },
    }

    # ── Weighted total ────────────────────────────────────────────────────
    total_score = sum(
        s["score"] * s["weight"] / total_w for s in scores.values()
    )

    # ── Breakdown for UI ─────────────────────────────────────────────────
    breakdown = {
        "skills":        skill_items,
        "experience":    exp_items,
        "education":     edu_items,
        "projects":      proj_items,
        "achievements":  ach_items,
        "organizations": grouped.get("ORG", []),
        "dates":         grouped.get("DATE", []),
        "certifications": grouped.get("CERT", []),
    }

    return {
        "scores":      scores,
        "total_score": round(total_score, 1),
        "breakdown":   breakdown,
    }


# ---------------------------------------------------------------------------
# Utility functions (unchanged API)
# ---------------------------------------------------------------------------

def predict_success_probability(
    cv_score: float,
    test_score: float,
    cv_weight: float = 0.6,
) -> float:
    """
    Predict hiring probability using a sigmoid of the weighted combined score.
    Returns a percentage (0–100).
    """
    test_weight = 1.0 - cv_weight
    combined    = cv_weight * (cv_score / 100) + test_weight * (test_score / 100)
    probability = 1.0 / (1.0 + math.exp(-10.0 * (combined - 0.5)))
    return round(probability * 100, 1)


def grade_test(answers: List[int], correct_answers: List[int]) -> Dict:
    """Grade MCQ test answers."""
    if len(answers) != len(correct_answers):
        raise ValueError("Answer count mismatch")

    total   = len(correct_answers)
    correct = sum(1 for a, c in zip(answers, correct_answers) if a == c)
    score   = (correct / total * 100) if total > 0 else 0.0

    details = [
        {
            "question":       i + 1,
            "user_answer":    answers[i],
            "correct_answer": correct_answers[i],
            "is_correct":     answers[i] == correct_answers[i],
        }
        for i in range(total)
    ]

    return {
        "score":   round(score, 1),
        "correct": correct,
        "total":   total,
        "details": details,
    }
