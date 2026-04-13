import type { WeightConfig, CvBreakdown, CvAnalysisResult, NerEntity, CategoryScore } from "./types"

// API endpoint - will proxy to FastAPI backend when running with Python Services
const API_BASE = "/api"

/**
 * Analyze CV using the BERT + BiLSTM + CRF model via API
 * Falls back to heuristic analysis if API is unavailable
 */
export async function analyzeCvWithDL(
  cvText: string,
  requiredSkills: string[],
  weights: WeightConfig
): Promise<{
  score: number
  breakdown: CvBreakdown
  entities: NerEntity[]
  modelInfo: { model_name: string; model_type: string }
}> {
  try {
    const response = await fetch(`${API_BASE}/analyze-cv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cv_text: cvText,
        required_skills: requiredSkills,
        weights: {
          skills: weights.skills,
          experience: weights.experience,
          projects: weights.projects,
          achievements: weights.achievements,
          education: weights.education,
        },
      }),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const result: CvAnalysisResult = await response.json()

    // Map API response to our format
    const breakdown: CvBreakdown = {
      skills: Math.round(result.scores.skills?.score ?? 0),
      experience: Math.round(result.scores.experience?.score ?? 0),
      projects: Math.round(result.scores.projects?.score ?? 0),
      achievements: Math.round(result.scores.achievements?.score ?? 0),
      education: Math.round(result.scores.education?.score ?? 0),
    }

    return {
      score: Math.round(result.total_score),
      breakdown,
      entities: result.entities,
      modelInfo: result.model_info,
    }
  } catch (error) {
    console.log("[v0] DL API unavailable, falling back to heuristic analysis:", error)
    // Fallback to heuristic analysis
    const { score, breakdown } = analyzeCv(cvText, requiredSkills, weights)
    return {
      score,
      breakdown,
      entities: [],
      modelInfo: { model_name: "heuristic", model_type: "Fallback (API unavailable)" },
    }
  }
}

// Keyword dictionaries for heuristic NLP scoring
const SKILL_KEYWORDS = [
  "javascript", "typescript", "python", "java", "c++", "react", "angular", "vue",
  "node", "express", "django", "flask", "sql", "nosql", "mongodb", "postgresql",
  "aws", "azure", "gcp", "docker", "kubernetes", "git", "ci/cd", "agile",
  "scrum", "rest", "graphql", "html", "css", "tailwind", "figma", "photoshop",
  "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
  "data analysis", "excel", "tableau", "power bi", "communication", "leadership",
  "teamwork", "problem solving", "critical thinking", "project management",
  "marketing", "sales", "finance", "accounting", "design", "ux", "ui",
  "blockchain", "solidity", "web3", "mobile", "ios", "android", "swift",
  "kotlin", "flutter", "dart", "rust", "go", "php", "laravel", "ruby",
  "rails", "spring", "microservices", "devops", "linux", "networking",
]

const EXPERIENCE_PATTERNS = [
  /(\d+)\+?\s*years?\s*(of\s+)?experience/gi,
  /experience[:\s]+(\d+)\+?\s*years?/gi,
  /worked\s+(for\s+)?(\d+)\+?\s*years?/gi,
  /(\d+)\+?\s*years?\s*(in|at|with)/gi,
  /intern(ship)?/gi,
  /senior|lead|principal|staff|junior|mid[\s-]level/gi,
  /manager|director|vp|chief|head\s+of/gi,
]

const PROJECT_PATTERNS = [
  /project[s]?[\s:]/gi,
  /built|developed|created|designed|implemented|architected/gi,
  /portfolio|github|open[\s-]source/gi,
  /hackathon|competition|challenge/gi,
  /freelance|contract|consulting/gi,
]

const ACHIEVEMENT_PATTERNS = [
  /award|prize|winner|first\s+place|recognition/gi,
  /certified|certification|license/gi,
  /published|paper|journal|conference/gi,
  /patent|invention/gi,
  /increased|improved|reduced|saved|generated|grew/gi,
  /\d+%\s*(increase|improvement|reduction|growth)/gi,
  /top\s+\d+%?/gi,
  /valedictorian|summa|magna|cum\s+laude/gi,
]

const EDUCATION_PATTERNS = [
  /bachelor|b\.?s\.?|b\.?a\.?|b\.?sc|b\.?eng/gi,
  /master|m\.?s\.?|m\.?a\.?|m\.?sc|m\.?eng|mba/gi,
  /ph\.?d|doctorate|doctoral/gi,
  /university|college|institute|school/gi,
  /gpa[\s:]+([0-9.]+)/gi,
  /degree|diploma|certificate/gi,
  /computer\s+science|engineering|business|mathematics|physics/gi,
]

function countMatches(text: string, patterns: RegExp[]): number {
  let count = 0
  for (const pattern of patterns) {
    const matches = text.match(new RegExp(pattern.source, pattern.flags))
    if (matches) count += matches.length
  }
  return count
}

function extractSkillScore(cvText: string, requiredSkills: string[]): number {
  const lowerCv = cvText.toLowerCase()
  let matchedSkills = 0
  let totalSkills = 0

  // Check required skills
  for (const skill of requiredSkills) {
    totalSkills++
    if (lowerCv.includes(skill.toLowerCase())) {
      matchedSkills++
    }
  }

  // Check general skill keywords
  let generalMatches = 0
  for (const keyword of SKILL_KEYWORDS) {
    if (lowerCv.includes(keyword)) {
      generalMatches++
    }
  }

  // FIX: was hardcoded 35 when no required skills — now rewards actual skill count
  const requiredScore = totalSkills > 0 ? (matchedSkills / totalSkills) * 70 : Math.min(generalMatches * 5, 65)
  // Additional skills beyond required get a bonus (rewards depth)
  const additionalBonus = totalSkills > 0 ? Math.min(Math.max(generalMatches - matchedSkills, 0) * 2, 30) : 0

  return Math.min(requiredScore + additionalBonus, 100)
}

function extractExperienceScore(cvText: string): number {
  const lowerCv = cvText.toLowerCase()
  let score = 0

  // 1. Explicit "X years of experience" mentions
  const yearMatches = cvText.match(/(\d+)\+?\s*years?/gi)
  let maxYears = 0
  if (yearMatches) {
    for (const match of yearMatches) {
      const num = parseInt(match)
      if (!isNaN(num) && num < 50) maxYears = Math.max(maxYears, num)
    }
  }
  score += Math.min(maxYears * 10, 30)

  // 2. Date range duration — handles "June 2024 – August 2024", "2022–2024", etc.
  const MONTH_MAP: Record<string, number> = {
    jan:1, feb:2, mar:3, apr:4, may:5, jun:6,
    jul:7, aug:8, sep:9, oct:10, nov:11, dec:12,
  }
  let totalMonths = 0

  // Named month ranges: "June 2024 – August 2024" or "June 2024 – Present"
  const namedRangeRe = /(?:\d{1,2}\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.,]*(\d{4})\s*[-\u2013\u2014to]+\s*(?:\d{1,2}\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)[a-z]*[\s.,]*(\d{4})?/gi
  let rm: RegExpExecArray | null
  while ((rm = namedRangeRe.exec(lowerCv)) !== null) {
    try {
      const y1 = parseInt(rm[2])
      const mon1 = MONTH_MAP[rm[1].slice(0,3)] ?? 1
      const isPresent = /present|current/.test(rm[3])
      const y2 = isPresent ? new Date().getFullYear() : (rm[4] ? parseInt(rm[4]) : y1)
      const mon2 = isPresent ? new Date().getMonth() + 1 : (MONTH_MAP[rm[3].slice(0,3)] ?? 12)
      totalMonths += Math.max((y2 - y1) * 12 + (mon2 - mon1), 0)
    } catch { /* skip malformed */ }
  }

  // Year-only ranges: "2022 – 2024" or "2022 – present"
  const yearRangeRe = /(20\d{2})\s*[-\u2013\u2014]\s*(20\d{2}|present|current)/gi
  while ((rm = yearRangeRe.exec(lowerCv)) !== null) {
    try {
      const y1 = parseInt(rm[1])
      const y2 = /present|current/.test(rm[2]) ? new Date().getFullYear() : parseInt(rm[2])
      // Only add if named range didn't already catch this period (avoid double-count)
      if (totalMonths === 0) totalMonths += Math.max((y2 - y1) * 12, 0)
    } catch { /* skip */ }
  }
  score += Math.min(totalMonths, 60) // 1 pt/month, max 60 pts (5 years)

  // 3. Internship / role count — CRITICAL for freshers
  // FIX: was using patternCount * 5 which gave ~10 pts for a fresher. Now gives a floor.
  const internshipCount = (lowerCv.match(/\binternship\b|\bintern\b/g) ?? []).length
  const roleEntries = (lowerCv.match(/(20\d{2})\s*[-\u2013\u2014]\s*(20\d{2}|present)/gi) ?? []).length
  const headerRoles = (lowerCv.match(/technical internship|other internship|work experience|summer internship/gi) ?? []).length
  const roles = Math.max(internshipCount, roleEntries, headerRoles)

  if (roles > 0) {
    score += Math.min(roles * 18, 36)
    score = Math.max(score, 45) // Floor: any real internship = at least 45
  }

  // 4. Seniority bonus
  if (/\b(director|principal|staff engineer)\b/i.test(cvText)) score += 25
  else if (/\b(senior|lead|manager)\b/i.test(cvText)) score += 15

  return Math.min(score, 100)
}

function extractProjectScore(cvText: string): number {
  const patternCount = countMatches(cvText, PROJECT_PATTERNS)
  let score = Math.min(patternCount * 8, 100)
  // Floor: if CV has multiple action verbs (built, developed, etc.) but
  // PROJECT_PATTERNS fired few times, give a minimum baseline
  const actionVerbCount = (cvText.match(/\b(built|developed|created|designed|implemented|deployed|engineered|automated)\b/gi) ?? []).length
  if (actionVerbCount >= 3 && score < 35) score = 35
  return score
}

function extractAchievementScore(cvText: string): number {
  const patternCount = countMatches(cvText, ACHIEVEMENT_PATTERNS)
  return Math.min(patternCount * 10, 100)
}

function extractEducationScore(cvText: string): number {
  const patternCount = countMatches(cvText, EDUCATION_PATTERNS)

  let bonus = 0
  const lowerCv = cvText.toLowerCase()
  if (lowerCv.match(/ph\.?d|doctorate/i)) bonus = 30
  else if (lowerCv.match(/master|m\.?s\.?|m\.?a\.?|mba/i)) bonus = 20
  else if (lowerCv.match(/bachelor|b\.?s\.?|b\.?a\.?/i)) bonus = 10

  // GPA bonus — detects multiple formats used in Indian/international CVs
  let gpaVal: number | null = null

  // Format 1: "GPA: 3.5"
  const gpaMatch1 = lowerCv.match(/gpa[\s:]+([0-9]+\.[0-9]+)/i)
  if (gpaMatch1) gpaVal = parseFloat(gpaMatch1[1])

  // Format 2: "3.39/4" (4-point scale)
  if (gpaVal === null) {
    const gpaMatch2 = lowerCv.match(/\b([0-9]+\.[0-9]+)\s*\/\s*4\b/)
    if (gpaMatch2) gpaVal = parseFloat(gpaMatch2[1])
  }

  // Format 3: "CGPA: 8.71" or "CGPA 8.71" (10-point scale — common in India)
  if (gpaVal === null) {
    const gpaMatch3 = lowerCv.match(/cgpa[\s:]+([0-9]+\.[0-9]+)/i)
    if (gpaMatch3) {
      const raw = parseFloat(gpaMatch3[1])
      gpaVal = raw > 4 ? (raw / 10) * 4 : raw  // normalise to /4
    }
  }

  // Format 4: "X.XX/10" notation
  if (gpaVal === null) {
    const gpaMatch4 = lowerCv.match(/\b([0-9]+\.[0-9]+)\s*\/\s*10\b/)
    if (gpaMatch4) gpaVal = (parseFloat(gpaMatch4[1]) / 10) * 4
  }

  if (gpaVal !== null) {
    if (gpaVal >= 3.7)      bonus += 20
    else if (gpaVal >= 3.5) bonus += 15
    else if (gpaVal >= 3.0) bonus += 10
    else if (gpaVal >= 2.5) bonus += 5
  }

  // Notable institution bonus
  if (/\b(iit|nit|bits|iiit|nmims|vit|mit|stanford|harvard)\b/i.test(lowerCv)) bonus += 10

  return Math.min(patternCount * 8 + bonus, 100)
}

export function analyzeCv(
  cvText: string,
  requiredSkills: string[],
  weights: WeightConfig
): { score: number; breakdown: CvBreakdown } {
  const breakdown: CvBreakdown = {
    skills: Math.round(extractSkillScore(cvText, requiredSkills)),
    experience: Math.round(extractExperienceScore(cvText)),
    projects: Math.round(extractProjectScore(cvText)),
    achievements: Math.round(extractAchievementScore(cvText)),
    education: Math.round(extractEducationScore(cvText)),
  }

  // Weighted score
  const totalWeight =
    weights.skills + weights.experience + weights.projects +
    weights.achievements + weights.education

  const weightedScore =
    (breakdown.skills * weights.skills +
      breakdown.experience * weights.experience +
      breakdown.projects * weights.projects +
      breakdown.achievements * weights.achievements +
      breakdown.education * weights.education) /
    (totalWeight || 1)

  return {
    score: Math.round(Math.min(Math.max(weightedScore, 0), 100)),
    breakdown,
  }
}

export function gradeTest(answers: number[], correctAnswers: number[]): number {
  if (correctAnswers.length === 0) return 100
  let correct = 0
  for (let i = 0; i < correctAnswers.length; i++) {
    if (answers[i] === correctAnswers[i]) correct++
  }
  return Math.round((correct / correctAnswers.length) * 100)
}

// Sigmoid-based probability prediction
export function predictSuccess(cvScore: number, testScore: number): number {
  const normalizedCv = cvScore / 100
  const normalizedTest = testScore / 100
  const combined = 0.6 * normalizedCv + 0.4 * normalizedTest
  // Sigmoid: steeper curve centered around 0.5
  const sigmoid = 1 / (1 + Math.exp(-10 * (combined - 0.5)))
  return Math.round(sigmoid * 100)
}

export function getMotivationalMessage(score: number): string {
  if (score >= 90) return "Outstanding performance! You absolutely crushed it!"
  if (score >= 80) return "Excellent work! You're a top-tier candidate!"
  if (score >= 70) return "Great job! You're looking very strong!"
  if (score >= 60) return "Good effort! Solid performance overall!"
  if (score >= 50) return "Not bad! There's room for improvement!"
  if (score >= 40) return "Keep going! Every experience counts!"
  return "Don't give up! Keep learning and growing!"
}
