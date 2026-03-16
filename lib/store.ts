import type { Job, Question, Application } from "./types"
import { SAMPLE_JOBS, SAMPLE_QUESTIONS } from "./sample-data"

const JOBS_KEY = "hiresense_jobs"
const QUESTIONS_KEY = "hiresense_questions"
const APPLICATIONS_KEY = "hiresense_applications"
const SEEDED_KEY = "hiresense_seeded"

function getItem<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback
  try {
    const item = localStorage.getItem(key)
    return item ? JSON.parse(item) : fallback
  } catch {
    return fallback
  }
}

function setItem<T>(key: string, value: T): void {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Storage full or other error
  }
}

export function seedDataIfNeeded(): void {
  if (typeof window === "undefined") return
  if (localStorage.getItem(SEEDED_KEY)) return

  setItem(JOBS_KEY, SAMPLE_JOBS)

  const allQuestions: Question[] = []
  for (const questions of Object.values(SAMPLE_QUESTIONS)) {
    allQuestions.push(...questions)
  }
  setItem(QUESTIONS_KEY, allQuestions)
  setItem(APPLICATIONS_KEY, [])
  localStorage.setItem(SEEDED_KEY, "true")
}

// Jobs
export function getJobs(): Job[] {
  return getItem<Job[]>(JOBS_KEY, [])
}

export function getJob(id: string): Job | undefined {
  return getJobs().find((j) => j.id === id)
}

export function createJob(job: Job): void {
  const jobs = getJobs()
  jobs.push(job)
  setItem(JOBS_KEY, jobs)
}

export function updateJob(id: string, updates: Partial<Job>): void {
  const jobs = getJobs().map((j) => (j.id === id ? { ...j, ...updates } : j))
  setItem(JOBS_KEY, jobs)
}

export function deleteJob(id: string): void {
  setItem(
    JOBS_KEY,
    getJobs().filter((j) => j.id !== id)
  )
  // Also delete related questions and applications
  setItem(
    QUESTIONS_KEY,
    getQuestions().filter((q) => q.jobId !== id)
  )
  setItem(
    APPLICATIONS_KEY,
    getApplications().filter((a) => a.jobId !== id)
  )
}

// Questions
export function getQuestions(): Question[] {
  return getItem<Question[]>(QUESTIONS_KEY, [])
}

export function getQuestionsForJob(jobId: string): Question[] {
  return getQuestions().filter((q) => q.jobId === jobId)
}

export function addQuestion(question: Question): void {
  const questions = getQuestions()
  questions.push(question)
  setItem(QUESTIONS_KEY, questions)
}

export function updateQuestion(id: string, updates: Partial<Question>): void {
  const questions = getQuestions().map((q) =>
    q.id === id ? { ...q, ...updates } : q
  )
  setItem(QUESTIONS_KEY, questions)
}

export function deleteQuestion(id: string): void {
  setItem(
    QUESTIONS_KEY,
    getQuestions().filter((q) => q.id !== id)
  )
}

// Applications
export function getApplications(): Application[] {
  return getItem<Application[]>(APPLICATIONS_KEY, [])
}

export function getApplicationsForJob(jobId: string): Application[] {
  return getApplications().filter((a) => a.jobId === jobId)
}

export function getApplicationsForStudent(studentId: string): Application[] {
  return getApplications().filter((a) => a.studentId === studentId)
}

export function getApplication(
  studentId: string,
  jobId: string
): Application | undefined {
  return getApplications().find(
    (a) => a.studentId === studentId && a.jobId === jobId
  )
}

export function createApplication(application: Application): void {
  const apps = getApplications()
  apps.push(application)
  setItem(APPLICATIONS_KEY, apps)
}
