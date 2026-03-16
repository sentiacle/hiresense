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

  const requiredScore = totalSkills > 0 ? (matchedSkills / totalSkills) * 70 : 35
  const generalScore = Math.min(generalMatches * 3, 30)

  return Math.min(requiredScore + generalScore, 100)
}

function extractExperienceScore(cvText: string): number {
  const yearMatches = cvText.match(/(\d+)\+?\s*years?/gi)
  let maxYears = 0
  if (yearMatches) {
    for (const match of yearMatches) {
      const num = parseInt(match)
      if (!isNaN(num) && num < 50) maxYears = Math.max(maxYears, num)
    }
  }

  const patternCount = countMatches(cvText, EXPERIENCE_PATTERNS)
  const yearsScore = Math.min(maxYears * 10, 50)
  const patternScore = Math.min(patternCount * 5, 50)

  return Math.min(yearsScore + patternScore, 100)
}

function extractProjectScore(cvText: string): number {
  const patternCount = countMatches(cvText, PROJECT_PATTERNS)
  return Math.min(patternCount * 8, 100)
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

  // GPA bonus
  const gpaMatch = lowerCv.match(/gpa[\s:]+([0-9.]+)/i)
  if (gpaMatch) {
    const gpa = parseFloat(gpaMatch[1])
    if (gpa >= 3.5) bonus += 15
    else if (gpa >= 3.0) bonus += 10
    else if (gpa >= 2.5) bonus += 5
  }

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
