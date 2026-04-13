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

// ---------------------------------------------------------------------------
// PDF text extraction — robust multi-strategy pipeline.
//
// Strategy 1a: PDF.js with version-correct CDN worker (handles compressed
//              streams, CID fonts — works on all modern text PDFs).
// Strategy 1b: PDF.js with fake worker (runs in main thread, no CDN needed).
// Strategy 2:  Tj/TJ operator regex on raw bytes (uncompressed PDFs only).
// Strategy 3:  Printable ASCII run extraction with noise filtering (last resort).
//
// Bugs fixed vs previous version:
//   • BUG 1 (CRITICAL): PDF.js returning "" caused an early `return ""` that
//     prevented Strategies 2–3 from running at all. Fixed: only return from
//     PDF.js when extracted text is substantial (≥ 200 chars).
//   • BUG 2 (CRITICAL): CDN URL used `.js` for all pdfjs versions, but
//     pdfjs-dist v4.x ships `.mjs`. Fixed: detect major version and use the
//     correct extension; fall back to unpkg if cdnjs doesn't have it.
//   • BUG 3 (HIGH): Strategy 2 regex relied on plain-text Tj operators which
//     are absent in FlateDecode-compressed streams (all NMIMS/Word PDFs).
//     Fixed: added Strategy 3 as a further fallback, and made Strategy 2 more
//     tolerant of escape sequences.
//   • BUG 4 (LOW): Threshold of 50 chars could pass binary garbage. Raised
//     to 200 and added a printable-char ratio sanity check.
// ---------------------------------------------------------------------------

/** Return true when the extracted string looks like real human text. */
function _isUsableText(text: string): boolean {
  if (text.length < 200) return false
  // At least 60 % of characters must be printable ASCII (space–tilde range)
  const printable = (text.match(/[ -~]/g) ?? []).length
  return printable / text.length >= 0.60
}

/** Run PDF.js extraction given an already-configured pdfjs module. */
async function _runPdfjsExtraction(
  pdfjs: typeof import("pdfjs-dist"),
  arrayBuffer: ArrayBuffer,
): Promise<string> {
  const pdf = await pdfjs.getDocument({
    data: new Uint8Array(arrayBuffer),
    useSystemFonts: true,   // better handling of embedded fonts
    disableFontFace: true,  // skip font rendering — we only need text
  }).promise

  const pageTexts: string[] = []

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum)
    const content = await page.getTextContent()
    const words: string[] = []

    for (const item of content.items) {
      if ("str" in item && typeof (item as any).str === "string") {
        const s = (item as any).str as string
        if (s.trim()) words.push(s)
        if ((item as any).hasEOL) words.push(" ")
      }
    }

    pageTexts.push(words.join(" "))
  }

  return pageTexts.join("\n").replace(/\s+/g, " ").trim()
}

async function extractTextFromPdf(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer()

  // ── Strategy 1a: PDF.js with version-aware CDN worker ───────────────────
  // pdfjs-dist v3.x  →  pdf.worker.min.js  (cdnjs has it)
  // pdfjs-dist v4.x+ →  pdf.worker.min.mjs (use unpkg; cdnjs often missing)
  try {
    const pdfjs = await import("pdfjs-dist")
    const majorVer = parseInt((pdfjs.version ?? "3").split(".")[0], 10)

    if (majorVer >= 4) {
      // v4.x: ES-module worker — unpkg is the most reliable source
      pdfjs.GlobalWorkerOptions.workerSrc =
        `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`
    } else {
      // v3.x: CJS worker — cdnjs works fine
      pdfjs.GlobalWorkerOptions.workerSrc =
        `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`
    }

    const text = await _runPdfjsExtraction(pdfjs, arrayBuffer)
    if (_isUsableText(text)) return text

    console.warn(
      `[HireSense] PDF.js (CDN worker v${pdfjs.version}) returned short/empty text. Trying fake worker.`
    )
  } catch (e) {
    console.warn("[HireSense] PDF.js CDN worker strategy failed:", e)
  }

  // ── Strategy 1b: PDF.js with fake worker (runs in main thread) ───────────
  // Avoids all CDN/CORS/version-mismatch issues. Slower but always works.
  try {
    const pdfjs = await import("pdfjs-dist")
    // Empty string tells pdfjs to use its built-in FakeWorker (main-thread).
    pdfjs.GlobalWorkerOptions.workerSrc = ""

    const text = await _runPdfjsExtraction(pdfjs, arrayBuffer)
    if (_isUsableText(text)) return text

    console.warn("[HireSense] PDF.js fake worker also returned short text.")
  } catch (e) {
    console.warn("[HireSense] PDF.js fake worker strategy failed:", e)
  }

  // ── Strategy 2: Raw Tj / TJ operator extraction ──────────────────────────
  // Works on uncompressed or partially-compressed PDFs.
  // Will silently produce nothing on FlateDecode-only PDFs — that is fine.
  try {
    const raw = new TextDecoder("latin1").decode(new Uint8Array(arrayBuffer))
    const parts: string[] = []

    // (text) Tj  — single string show
    for (const m of raw.matchAll(/\(([^()]{1,300})\)\s*Tj/g)) {
      const s = m[1]
        .replace(/\\n/g, " ").replace(/\\r/g, " ")
        .replace(/\\t/g, " ").replace(/\\\\/g, "\\")
        .replace(/\\\(/g, "(").replace(/\\\)/g, ")")
        .trim()
      if (s.length > 1) parts.push(s)
    }

    // [(text)(text)...] TJ  — array show
    for (const m of raw.matchAll(/\[([^\]]{1,800})\]\s*TJ/g)) {
      for (const n of m[1].matchAll(/\(([^()]{1,200})\)/g)) {
        const s = n[1].replace(/\\n/g, " ").trim()
        if (s.length > 1) parts.push(s)
      }
    }

    const combined = parts.join(" ").replace(/\s+/g, " ").trim()
    if (_isUsableText(combined)) return combined
  } catch (e) {
    console.warn("[HireSense] Strategy 2 (Tj/TJ) failed:", e)
  }

  // ── Strategy 3: Printable ASCII run extraction ───────────────────────────
  // Last resort. Grabs long runs of printable characters, strips PDF keywords.
  try {
    const raw = new TextDecoder("latin1").decode(new Uint8Array(arrayBuffer))

    // PDF internal keywords that should never appear in extracted text
    const PDF_OPS = new Set([
      "obj","endobj","stream","endstream","xref","trailer","startxref",
      "BT","ET","Td","TD","Tf","Tm","TL","Tr","Ts","Tz","Tc","Tw","TJ","Tj",
      "Do","cm","re","gs","cs","CS","sc","SC","rg","RG","Type","Page","Font",
      "Resources","MediaBox","Contents","Kids","Root","Catalog","FlateDecode",
      "DCTDecode","Filter","Length","Subtype","Encoding","Widths","BaseFont",
    ])

    const runs = (raw.match(/[ !-~]{5,}/g) ?? [])
      .filter(s => {
        const trimmed = s.trim()
        if (PDF_OPS.has(trimmed)) return false
        // Must contain at least one alphabetic character
        if (!/[a-zA-Z]/.test(trimmed)) return false
        // Reject strings that look like hex or binary noise
        if (/^[0-9a-f\s]{8,}$/i.test(trimmed)) return false
        return true
      })

    const text = runs.join(" ").replace(/\s+/g, " ").trim()
    if (_isUsableText(text)) return text
  } catch (e) {
    console.warn("[HireSense] Strategy 3 (ASCII runs) failed:", e)
  }

  // All strategies exhausted
  return ""
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
  const [isExtracting, setIsExtracting] = useState(false)

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
    handleSubmitAnswer(-1)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentQuestion])

  const timer = useTimer(currentTimeLimit, handleTimerExpire)

  // FIX: replaced broken regex extractor with PDF.js-based extraction
  const handlePdfSelect = async (file: File | null) => {
    if (!file) return
    if (file.type !== "application/pdf") {
      toast.error("Please upload a PDF file")
      return
    }

    setIsExtracting(true)
    toast.info("Reading your CV...", { duration: 2000 })

    try {
      const extracted = await extractTextFromPdf(file)

      if (!extracted || extracted.length < 200) {
        toast.error(
          "Could not read enough text from this PDF. " +
          "Make sure it is a text-based PDF (not a scan). " +
          "If the issue persists, copy-paste your CV text instead.",
          { duration: 8000 }
        )
        setIsExtracting(false)
        return
      }

      setCvText(extracted)
      setCvFileName(file.name)

      // Also store data URL for download by recruiter
      const reader = new FileReader()
      reader.onload = () => {
        setCvFileDataUrl(typeof reader.result === "string" ? reader.result : "")
      }
      reader.readAsDataURL(file)

      toast.success(`CV read successfully (${extracted.length} characters extracted)`)
    } catch (err) {
      toast.error("Failed to read PDF. Please try another file.")
      console.error("[HireSense] PDF extraction error:", err)
    } finally {
      setIsExtracting(false)
    }
  }

  const handleCvUpload = () => {
    if (!cvText.trim() || !cvFileName) {
      toast.error("Please upload your CV PDF first")
      return
    }

    fire()
    toast.success("CV uploaded! Good luck :)", {
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
      void submitApplication(cvText, cvFileName, [])
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
        timer.pause()
        fire()
        toast.success("Quiz completed! Calculating your results...", {
          duration: 3000,
          style: {
            background: "oklch(0.5 0.2 270)",
            color: "white",
            border: "none",
            fontSize: "14px",
            fontWeight: "600",
          },
        })
        setTimeout(() => {
          void submitApplication(cvText, cvFileName, newAnswers)
        }, 1500)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [answers, currentQuestion, questions, cvText, cvFileName]
  )

  const submitApplication = async (
    text: string,
    fileName: string,
    testAnswers: number[]
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
      cvFileDataUrl,
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
        ].map((s, i, arr) => (
          <div key={s.key} className="flex items-center gap-2">
            <div
              className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                step === s.key
                  ? "bg-primary text-primary-foreground"
                  : ["upload", "quiz", "results"].indexOf(step) > i
                  ? "bg-success/20 text-success"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              <s.icon className="h-4 w-4" />
              {s.label}
            </div>
            {i < arr.length - 1 && (
              <div className="h-px w-8 bg-border" />
            )}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {/* STEP 1: Upload CV */}
        {step === "upload" && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="max-w-2xl"
          >
            <div className="rounded-xl border border-border/60 bg-card p-8">
              <h2 className="mb-2 text-xl font-semibold text-foreground">
                Upload Your CV
              </h2>
              <p className="mb-6 text-sm text-muted-foreground">
                Upload a text-based PDF CV. The AI will analyze it and score it
                against the job requirements.
              </p>

              <label
                htmlFor="cv-upload"
                className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 transition-colors ${
                  cvFileName
                    ? "border-success/50 bg-success/5"
                    : "border-border hover:border-primary/50 hover:bg-primary/5"
                }`}
              >
                {cvFileName ? (
                  <>
                    <FileText className="h-12 w-12 text-success" />
                    <p className="text-sm font-semibold text-success">
                      {cvFileName}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {cvText.length} characters extracted — click to replace
                    </p>
                  </>
                ) : isExtracting ? (
                  <>
                    <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                    <p className="text-sm text-muted-foreground">Reading PDF...</p>
                  </>
                ) : (
                  <>
                    <Upload className="h-12 w-12 text-muted-foreground/50" />
                    <p className="text-sm font-medium text-foreground">
                      Click to upload your CV (PDF)
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Use a text-based PDF (not a scanned image)
                    </p>
                  </>
                )}
                <input
                  id="cv-upload"
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => handlePdfSelect(e.target.files?.[0] ?? null)}
                  disabled={isExtracting}
                />
              </label>

              <motion.button
                onClick={handleCvUpload}
                disabled={!cvFileName || isExtracting}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3.5 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
              >
                Continue to Assessment
                <ArrowRight className="h-5 w-5" />
              </motion.button>
            </div>
          </motion.div>
        )}

        {/* STEP 2: Quiz */}
        {step === "quiz" && questions.length > 0 && (
          <motion.div
            key="quiz"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="max-w-2xl"
          >
            <div className="mb-4 flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Question {currentQuestion + 1} of {questions.length}
              </span>
              <span
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-bold ${
                  timer.timeLeft <= 10
                    ? "bg-destructive/10 text-destructive"
                    : "bg-primary/10 text-primary"
                }`}
              >
                <Timer className="h-3.5 w-3.5" />
                {timer.timeLeft}s
              </span>
            </div>

            <div className="h-1.5 overflow-hidden rounded-full bg-muted mb-6">
              <motion.div
                animate={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                className="h-full rounded-full bg-primary"
                transition={{ duration: 0.3 }}
              />
            </div>

            <div className="rounded-xl border border-border/60 bg-card p-8">
              <p className="mb-6 text-lg font-semibold text-foreground">
                {questions[currentQuestion]?.question}
              </p>

              <div className="space-y-3">
                {questions[currentQuestion]?.options.map((option, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedOption(i)}
                    className={`w-full rounded-xl border-2 p-4 text-left text-sm font-medium transition-all ${
                      selectedOption === i
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-background text-foreground hover:border-primary/40 hover:bg-primary/5"
                    }`}
                  >
                    <span className="mr-3 inline-flex h-6 w-6 items-center justify-center rounded-full border border-current text-xs">
                      {String.fromCharCode(65 + i)}
                    </span>
                    {option}
                  </button>
                ))}
              </div>

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
                <p className="mb-2 text-sm font-medium text-muted-foreground">CV Score</p>
                <div className="mb-1 text-4xl font-bold text-foreground">
                  <AnimatedCounter value={application.cvScore} suffix="/100" />
                </div>
                <p className="text-xs text-muted-foreground">Requirement status shown below</p>
                <div className="mt-3">
                  {application.cvScore >= job.minCvScore ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                      <CheckCircle2 className="h-3 w-3" /> Passed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-semibold text-destructive">
                      <XCircle className="h-3 w-3" /> Below threshold
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
                <p className="mb-2 text-sm font-medium text-muted-foreground">Test Score</p>
                <div className="mb-1 text-4xl font-bold text-foreground">
                  <AnimatedCounter value={application.testScore} suffix="/100" />
                </div>
                <p className="text-xs text-muted-foreground">Requirement status shown below</p>
                <div className="mt-3">
                  {application.testScore >= job.minTestScore ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                      <CheckCircle2 className="h-3 w-3" /> Passed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-semibold text-destructive">
                      <XCircle className="h-3 w-3" /> Below threshold
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
                <GaugeChart value={application.probability} size={160} label="Job Match Probability" />
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
                <p className={`text-lg font-bold ${application.passed ? "text-success" : "text-destructive"}`}>
                  {application.passed ? "You Passed! Your application is strong." : "Did Not Meet Criteria This Time"}
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
              <h3 className="mb-5 text-lg font-semibold text-foreground">CV Score Breakdown</h3>
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
                      <span className="text-sm font-medium text-foreground">{factor.label}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-muted-foreground">Weight: {factor.weight}%</span>
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
