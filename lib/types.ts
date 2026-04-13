export interface User {
  id: string
  name: string
  email: string
  role: "recruiter" | "student"
}

export interface WeightConfig {
  skills: number
  experience: number
  projects: number
  achievements: number
  education: number
}

export interface Job {
  id: string
  title: string
  company: string
  description: string
  requiredSkills: string[]
  weights: WeightConfig
  minCvScore: number
  minTestScore: number
  recruiterId: string
  createdAt: string
}

export interface Question {
  id: string
  jobId: string
  question: string
  options: string[]
  correctAnswer: number
  timeLimit: number // seconds
}

export interface CvBreakdown {
  skills: number
  experience: number
  projects: number
  achievements: number
  education: number
}

// NER Entity from Deep Learning model
export interface NerEntity {
  text: string
  label: string
  start: number
  end: number
  confidence: number
}

// Category score with entities
export interface CategoryScore {
  score: number
  weight: number
  entities: string[]
  details: string
}

// Full CV analysis result from API
export interface CvAnalysisResult {
  success: boolean
  entities: NerEntity[]
  scores: Record<string, CategoryScore>
  total_score: number
  breakdown: Record<string, string[]>
  model_info: {
    model_name: string
    device: string
    model_type: string
  }
}

export interface Application {
  id: string
  studentId: string
  studentName: string
  studentEmail: string
  jobId: string
  cvFileName: string
  cvText: string
  cvFileDataUrl?: string
  cvScore: number
  testScore: number
  probability: number
  passed: boolean
  breakdown: CvBreakdown
  answers: number[]
  completedAt: string
}
