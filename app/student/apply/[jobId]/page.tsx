"use client"

import { useState, useCallback, useEffect } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import {
  ArrowLeft,
  Upload,
  FileText,
  Timer,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Sparkles,
} from "lucide-react"
import { getJob, getQuestionsForJob, getApplication, createApplication } from "@/lib/store"
import { useAuth } from "@/lib/auth-context"
import {
  analyzeCvWithDL,
  gradeTest,
  predictSuccess,
  getMotivationalMessage,
} from "@/lib/scoring"
import { useConfetti } from "@/hooks/use-confetti"
import { useTimer } from "@/hooks/use-timer"
import { AnimatedCounter } from "@/components/shared/animated-counter"
import { GaugeChart } from "@/components/shared/gauge-chart"
import { PageTransition } from "@/components/shared/page-transition"
import type { Application } from "@/lib/types"
import { toast } from "sonner"
import useSWR from "swr"

type Step = "upload" | "quiz" | "results"


function extractTextFromPdfBytes(bytes: Uint8Array): string {
  try {
    const raw = new TextDecoder("latin1").decode(bytes)
    const matches = [...raw.matchAll(/\(([^()]{2,})\)\s*Tj/g)]
    const extracted = matches
      .map((m) => m[1])
      .join(" ")
      .replace(/\\[nrtbf]/g, " ")
      .replace(/\s+/g, " ")
      .trim()

    if (extracted.length > 100) return extracted

    return raw
      .replace(/[^\x20-\x7E\n]/g, " ")
      .replace(/\s+/g, " ")
      .trim()
  } catch {
    return ""
  }
}

export default function ApplyPage() {
  const params = useParams()
  const jobId = params.jobId as string
  const { user } = useAuth()
  const { fire, fireSmall } = useConfetti()

  const { data } = useSWR(
    user ? `apply-${jobId}-${user.id}` : null,
    () => ({
      job: getJob(jobId),
      questions: getQuestionsForJob(jobId),
      existingApp: user ? getApplication(user.id, jobId) : undefined,
    })
  )

  const job = data?.job
  const questions = data?.questions ?? []
  const existingApp = data?.existingApp

  const [step, setStep] = useState<Step>("upload")
  const [cvText, setCvText] = useState("")
  const [cvFileName, setCvFileName] = useState("")
  const [cvFileDataUrl, setCvFileDataUrl] = useState("")
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<number[]>([])
  const [selectedOption, setSelectedOption] = useState<number | null>(null)
  const [application, setApplication] = useState<Application | null>(null)

  // If already applied, show results
  useEffect(() => {
    if (existingApp) {
      setApplication(existingApp)
      setStep("results")
    }
  }, [existingApp])

  // Timer for current question
  const currentTimeLimit = questions[currentQuestion]?.timeLimit ?? 30

  const handleTimerExpire = useCallback(() => {
    // Auto-submit with -1 (no answer)
    handleSubmitAnswer(-1)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentQuestion])

  const timer = useTimer(currentTimeLimit, handleTimerExpire)

  const handlePdfSelect = async (file: File | null) => {
    if (!file) return
    if (file.type !== "application/pdf") {
      toast.error("Please upload a PDF file")
      return
    }

    const bytes = new Uint8Array(await file.arrayBuffer())
    const extracted = extractTextFromPdfBytes(bytes)
    if (!extracted || extracted.length < 30) {
      toast.error("Could not read enough text from this PDF. Please upload another CV PDF.")
      return
    }

    setCvText(extracted)
    setCvFileName(file.name)

    const reader = new FileReader()
    reader.onload = () => {
      setCvFileDataUrl(typeof reader.result === "string" ? reader.result : "")
    }
    reader.readAsDataURL(file)
  }

  const handleCvUpload = () => {
    if (!cvText.trim() || !cvFileName) {
      toast.error("Please upload your CV PDF first")
      return
    }

    fire()
    toast.success("AMAZING! THE CV WAS UPLOADED! GOOD LUCK :)", {
      duration: 4000,
      style: {
        background: "oklch(0.5 0.2 270)",
        color: "white",
        border: "none",
        fontSize: "14px",
        fontWeight: "600",
      },
    })

    if (questions.length > 0) {
      setTimeout(() => {
        setStep("quiz")
        setAnswers([])
        setCurrentQuestion(0)
        timer.reset(questions[0]?.timeLimit ?? 30)
        timer.start()
      }, 1500)
    } else {
      void submitApplication(cvText, cvFileName, [], cvFileDataUrl)
    }
  }

  const handleSubmitAnswer = useCallback(
    (answer: number) => {
      const newAnswers = [...answers, answer === -1 ? -1 : answer]
      setAnswers(newAnswers)
      setSelectedOption(null)

      if (currentQuestion < questions.length - 1) {
        const nextQ = currentQuestion + 1
        setCurrentQuestion(nextQ)
        timer.reset(questions[nextQ]?.timeLimit ?? 30)
        timer.start()
      } else {
        // Quiz done
        timer.pause()
        fire()
        toast.success(
          "Quiz completed! Fingers crossed! Calculating your results...",
          {
            duration: 3000,
            style: {
              background: "oklch(0.5 0.2 270)",
              color: "white",
              border: "none",
              fontSize: "14px",
              fontWeight: "600",
            },
          }
        )
        setTimeout(() => {
          void submitApplication(cvText, cvFileName, newAnswers, cvFileDataUrl)
        }, 1500)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [answers, currentQuestion, questions, cvText, cvFileName]
  )

  const submitApplication = async (
    text: string,
    fileName: string,
    testAnswers: number[],
    fileDataUrl?: string
  ) => {
    if (!user || !job) return

    const {
      score: cvScore,
      breakdown,
      modelInfo,
    } = await analyzeCvWithDL(
      text,
      job.requiredSkills,
      job.weights
    )

    if (modelInfo.model_name !== "heuristic") {
      toast.success(`CV analyzed with ${modelInfo.model_type}`)
    }

    const correctAnswers = questions.map((q) => q.correctAnswer)
    const testScore = gradeTest(testAnswers, correctAnswers)
    const probability = predictSuccess(cvScore, testScore)
    const passed = cvScore >= job.minCvScore && testScore >= job.minTestScore

    const app: Application = {
      id: `app-${Date.now()}`,
      studentId: user.id,
      studentName: user.name,
      studentEmail: user.email,
      jobId: job.id,
      cvFileName: fileName,
      cvText: text,
      cvFileDataUrl: fileDataUrl,
      cvScore,
      testScore,
      probability,
      passed,
      breakdown,
      answers: testAnswers,
      completedAt: new Date().toISOString(),
    }

    createApplication(app)
    setApplication(app)
    setStep("results")

    setTimeout(() => {
      fireSmall()
    }, 500)
  }

  if (!job) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-muted-foreground">Job not found.</p>
      </div>
    )
  }

  return (
    <PageTransition>
      <div className="mb-8">
        <Link
          href="/student"
          className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Jobs
        </Link>
        <h1 className="text-3xl font-bold text-foreground">
          Apply: {job.title}
        </h1>
        <p className="mt-1 text-muted-foreground">{job.company}</p>
      </div>

      {/* Step Indicator */}
      <div className="mb-8 flex items-center gap-4">
        {[
          { key: "upload", label: "Upload CV", icon: Upload },
          { key: "quiz", label: "Assessment", icon: Timer },
          { key: "results", label: "Results", icon: Sparkles },
        ].map((s, i) => (
          <div key={s.key} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={`h-0.5 w-8 ${
                  step === s.key || (step === "results" && i <= 2) || (step === "quiz" && i <= 1)
                    ? "bg-primary"
                    : "bg-border"
                }`}
              />
            )}
            <div
              className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium ${
                step === s.key
                  ? "bg-primary text-primary-foreground"
                  : step === "results" || (step === "quiz" && s.key === "upload")
                  ? "bg-primary/10 text-primary"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              <s.icon className="h-4 w-4" />
              {s.label}
            </div>
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {/* STEP 1: Upload CV */}
        {step === "upload" && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="max-w-3xl"
          >
            <div className="rounded-xl border border-border/60 bg-card p-8">
              <div className="mb-6 flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                  <FileText className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-foreground">
                    Upload Your CV
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Upload your CV PDF below for AI analysis.
                  </p>
                </div>
              </div>

              {/* Drag area visual + textarea */}
              <div className="mb-4">
                <div className="rounded-xl border-2 border-dashed border-primary/20 bg-primary/5 p-6 text-center transition-colors hover:border-primary/40">
                  <Upload className="mx-auto mb-3 h-10 w-10 text-primary/40" />
                  <p className="mb-1 text-sm font-medium text-foreground">
                    Upload your CV PDF
                  </p>
                  <p className="text-xs text-muted-foreground">
                    PDF upload is required. We extract text from your uploaded CV for scoring
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-input bg-background p-4">
                <label
                  htmlFor="cv-pdf"
                  className="mb-2 block text-sm font-medium text-foreground"
                >
                  Upload CV (PDF)
                </label>
                <input
                  id="cv-pdf"
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => void handlePdfSelect(e.target.files?.[0] ?? null)}
                  className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground"
                />
                {cvFileName && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Selected: {cvFileName}
                  </p>
                )}
              </div>

              <motion.button
                onClick={handleCvUpload}
                disabled={!cvText.trim() || !cvFileName}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3.5 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
              >
                <Upload className="h-5 w-5" />
                Upload CV & {questions.length > 0 ? "Start Assessment" : "Get Score"}
              </motion.button>
            </div>

            {/* Job details */}
            <div className="mt-6 rounded-xl border border-border/60 bg-card p-6">
              <h3 className="mb-3 text-lg font-semibold text-foreground">
                About This Position
              </h3>
              <p className="mb-4 text-sm text-muted-foreground leading-relaxed">
                {job.description}
              </p>
              <div className="flex flex-wrap gap-2">
                {job.requiredSkills.map((skill) => (
                  <span
                    key={skill}
                    className="rounded-md bg-primary/8 px-2.5 py-0.5 text-xs font-medium text-primary"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* STEP 2: Quiz */}
        {step === "quiz" && questions.length > 0 && (
          <motion.div
            key="quiz"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="max-w-3xl"
          >
            {/* Progress */}
            <div className="mb-6">
              <div className="mb-2 flex items-center justify-between text-sm">
                <span className="font-medium text-foreground">
                  Question {currentQuestion + 1} of {questions.length}
                </span>
                <span className="text-muted-foreground">
                  {Math.round(((currentQuestion) / questions.length) * 100)}%
                  complete
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: 0 }}
                  animate={{
                    width: `${((currentQuestion) / questions.length) * 100}%`,
                  }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>

            <div className="rounded-xl border border-border/60 bg-card p-8">
              {/* Timer */}
              <div className="mb-6 flex items-center justify-between">
                <span className="text-sm font-medium text-muted-foreground">
                  Time remaining
                </span>
                <div
                  className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-bold ${
                    timer.secondsLeft <= 5
                      ? "bg-destructive/10 text-destructive animate-pulse"
                      : timer.secondsLeft <= 10
                      ? "bg-warning/10 text-warning-foreground"
                      : "bg-primary/10 text-primary"
                  }`}
                >
                  <Timer className="h-4 w-4" />
                  {timer.secondsLeft}s
                </div>
              </div>

              {/* Timer bar */}
              <div className="mb-6 h-1.5 overflow-hidden rounded-full bg-muted">
                <motion.div
                  className={`h-full rounded-full transition-colors ${
                    timer.secondsLeft <= 5
                      ? "bg-destructive"
                      : timer.secondsLeft <= 10
                      ? "bg-accent"
                      : "bg-primary"
                  }`}
                  style={{ width: `${timer.percentage}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>

              {/* Question */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentQuestion}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.25 }}
                >
                  <h2 className="mb-6 text-xl font-semibold text-foreground">
                    {questions[currentQuestion]?.question}
                  </h2>

                  <div className="space-y-3">
                    {questions[currentQuestion]?.options.map((opt, i) => (
                      <button
                        key={i}
                        onClick={() => setSelectedOption(i)}
                        className={`flex w-full items-center gap-3 rounded-xl border-2 px-5 py-4 text-left text-sm transition-all ${
                          selectedOption === i
                            ? "border-primary bg-primary/5 shadow-md shadow-primary/10"
                            : "border-border bg-background hover:border-primary/30"
                        }`}
                      >
                        <div
                          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                            selectedOption === i
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {String.fromCharCode(65 + i)}
                        </div>
                        <span
                          className={
                            selectedOption === i
                              ? "font-medium text-foreground"
                              : "text-foreground"
                          }
                        >
                          {opt}
                        </span>
                      </button>
                    ))}
                  </div>
                </motion.div>
              </AnimatePresence>

              <motion.button
                onClick={() => {
                  if (selectedOption !== null) {
                    handleSubmitAnswer(selectedOption)
                  }
                }}
                disabled={selectedOption === null}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3.5 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
              >
                {currentQuestion < questions.length - 1 ? (
                  <>
                    Next Question
                    <ArrowRight className="h-5 w-5" />
                  </>
                ) : (
                  <>
                    Submit & See Results
                    <Sparkles className="h-5 w-5" />
                  </>
                )}
              </motion.button>
            </div>
          </motion.div>
        )}

        {/* STEP 3: Results */}
        {step === "results" && application && (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Motivational banner */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="mb-8 rounded-2xl bg-gradient-to-r from-primary/10 via-primary/5 to-accent/10 p-8 text-center"
            >
              <Sparkles className="mx-auto mb-3 h-10 w-10 text-primary" />
              <h2 className="mb-2 text-2xl font-bold text-foreground">
                {getMotivationalMessage(application.probability)}
              </h2>
              <p className="text-muted-foreground">
                {application.passed
                  ? "Congratulations! You've met the passing criteria for this position!"
                  : "Keep working hard! Every application is a learning experience."}
              </p>
            </motion.div>

            {/* Score Cards */}
            <div className="mb-8 grid gap-5 md:grid-cols-3">
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="rounded-xl border border-border/60 bg-card p-6 text-center"
              >
                <p className="mb-2 text-sm font-medium text-muted-foreground">
                  CV Score
                </p>
                <div className="mb-1 text-4xl font-bold text-foreground">
                  <AnimatedCounter value={application.cvScore} suffix="/100" />
                </div>
                <p className="text-xs text-muted-foreground">
                  Requirement status shown below
                </p>
                <div className="mt-3">
                  {application.cvScore >= job.minCvScore ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                      <CheckCircle2 className="h-3 w-3" />
                      Passed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-semibold text-destructive">
                      <XCircle className="h-3 w-3" />
                      Below threshold
                    </span>
                  )}
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="rounded-xl border border-border/60 bg-card p-6 text-center"
              >
                <p className="mb-2 text-sm font-medium text-muted-foreground">
                  Test Score
                </p>
                <div className="mb-1 text-4xl font-bold text-foreground">
                  <AnimatedCounter value={application.testScore} suffix="/100" />
                </div>
                <p className="text-xs text-muted-foreground">
                  Requirement status shown below
                </p>
                <div className="mt-3">
                  {application.testScore >= job.minTestScore ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                      <CheckCircle2 className="h-3 w-3" />
                      Passed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-semibold text-destructive">
                      <XCircle className="h-3 w-3" />
                      Below threshold
                    </span>
                  )}
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="flex flex-col items-center rounded-xl border border-border/60 bg-card p-6"
              >
                <GaugeChart
                  value={application.probability}
                  size={160}
                  label="Job Match Probability"
                />
              </motion.div>
            </div>

            {/* Overall Status */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className={`mb-8 flex items-center gap-4 rounded-xl border p-5 ${
                application.passed
                  ? "border-success/30 bg-success/5"
                  : "border-destructive/30 bg-destructive/5"
              }`}
            >
              {application.passed ? (
                <CheckCircle2 className="h-8 w-8 shrink-0 text-success" />
              ) : (
                <XCircle className="h-8 w-8 shrink-0 text-destructive" />
              )}
              <div>
                <p
                  className={`text-lg font-bold ${
                    application.passed ? "text-success" : "text-destructive"
                  }`}
                >
                  {application.passed
                    ? "You Passed! Your application is strong."
                    : "Did Not Meet Criteria This Time"}
                </p>
                <p className="text-sm text-muted-foreground">
                  {application.passed
                    ? "Your scores met all configured requirements for this role."
                    : "You did not meet all configured requirements this time."}
                </p>
              </div>
            </motion.div>

            {/* CV Breakdown */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="rounded-xl border border-border/60 bg-card p-6"
            >
              <h3 className="mb-5 text-lg font-semibold text-foreground">
                CV Score Breakdown
              </h3>
              <div className="space-y-4">
                {[
                  { label: "Skills", value: application.breakdown.skills, weight: job.weights.skills },
                  { label: "Experience", value: application.breakdown.experience, weight: job.weights.experience },
                  { label: "Projects", value: application.breakdown.projects, weight: job.weights.projects },
                  { label: "Achievements", value: application.breakdown.achievements, weight: job.weights.achievements },
                  { label: "Education", value: application.breakdown.education, weight: job.weights.education },
                ].map((factor, i) => (
                  <div key={factor.label}>
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">
                        {factor.label}
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-muted-foreground">
                          Weight: {factor.weight}%
                        </span>
                        <span className="min-w-[2.5rem] text-right text-sm font-bold text-primary">
                          {factor.value}
                        </span>
                      </div>
                    </div>
                    <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${factor.value}%` }}
                        transition={{ duration: 0.8, delay: 0.8 + i * 0.1 }}
                        className="h-full rounded-full bg-primary"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            <div className="mt-6">
              <Link
                href="/student"
                className="inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Jobs
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </PageTransition>
  )
}
