"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import {
  ArrowLeft,
  Plus,
  Trash2,
  Timer,
  CheckCircle2,
  AlertCircle,
} from "lucide-react"
import { getJob, getQuestionsForJob, addQuestion, deleteQuestion } from "@/lib/store"
import type { Question } from "@/lib/types"
import { toast } from "sonner"
import { PageTransition } from "@/components/shared/page-transition"
import useSWR from "swr"

export default function QuestionsPage() {
  const params = useParams()
  const jobId = params.jobId as string

  const { data, mutate } = useSWR(`questions-${jobId}`, () => ({
    job: getJob(jobId),
    questions: getQuestionsForJob(jobId),
  }))

  const job = data?.job
  const questions = data?.questions ?? []

  const [questionText, setQuestionText] = useState("")
  const [options, setOptions] = useState(["", "", "", ""])
  const [correctAnswer, setCorrectAnswer] = useState(0)
  const [timeLimit, setTimeLimit] = useState(30)
  const [showForm, setShowForm] = useState(false)

  const updateOption = (index: number, value: string) => {
    const newOptions = [...options]
    newOptions[index] = value
    setOptions(newOptions)
  }

  const handleAddQuestion = (e: React.FormEvent) => {
    e.preventDefault()
    if (!questionText || options.some((o) => !o.trim())) {
      toast.error("Please fill in all fields")
      return
    }

    const q: Question = {
      id: `q-${Date.now()}`,
      jobId,
      question: questionText,
      options: options.map((o) => o.trim()),
      correctAnswer,
      timeLimit,
    }

    addQuestion(q)
    toast.success("Question added!")
    setQuestionText("")
    setOptions(["", "", "", ""])
    setCorrectAnswer(0)
    setTimeLimit(30)
    setShowForm(false)
    mutate()
  }

  const handleDelete = (id: string) => {
    deleteQuestion(id)
    toast.success("Question removed")
    mutate()
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
          href="/recruiter"
          className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>
        <h1 className="text-3xl font-bold text-foreground">
          Questions for {job.title}
        </h1>
        <p className="mt-1 text-muted-foreground">
          Add timed MCQ questions that candidates will answer during their
          assessment.
        </p>
      </div>

      {/* Questions List */}
      <div className="mb-6 space-y-3">
        <AnimatePresence>
          {questions.map((q, i) => (
            <motion.div
              key={q.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="rounded-xl border border-border/60 bg-card p-5"
            >
              <div className="mb-3 flex items-start justify-between gap-4">
                <div className="flex-1">
                  <span className="mr-2 inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                    {i + 1}
                  </span>
                  <span className="text-sm font-medium text-foreground">
                    {q.question}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="flex items-center gap-1 rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground">
                    <Timer className="h-3 w-3" />
                    {q.timeLimit}s
                  </span>
                  <button
                    onClick={() => handleDelete(q.id)}
                    className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                    aria-label="Delete question"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {q.options.map((opt, oi) => (
                  <div
                    key={oi}
                    className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                      oi === q.correctAnswer
                        ? "bg-success/10 text-success font-medium"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {oi === q.correctAnswer ? (
                      <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                    ) : (
                      <span className="h-3.5 w-3.5 shrink-0 rounded-full border border-muted-foreground/30" />
                    )}
                    {opt}
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {questions.length === 0 && !showForm && (
          <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border py-12">
            <AlertCircle className="mb-3 h-10 w-10 text-muted-foreground/40" />
            <p className="mb-1 font-medium text-muted-foreground">
              No questions yet
            </p>
            <p className="text-sm text-muted-foreground">
              Add questions for candidates to answer during their assessment.
            </p>
          </div>
        )}
      </div>

      {/* Add Question Form */}
      {showForm ? (
        <motion.form
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          onSubmit={handleAddQuestion}
          className="max-w-3xl rounded-xl border border-primary/20 bg-card p-6"
        >
          <h3 className="mb-4 text-lg font-semibold text-foreground">
            New Question
          </h3>
          <div className="space-y-4">
            <div>
              <label htmlFor="qtext" className="mb-1.5 block text-sm font-medium text-foreground">
                Question
              </label>
              <textarea
                id="qtext"
                required
                rows={2}
                value={questionText}
                onChange={(e) => setQuestionText(e.target.value)}
                placeholder="Enter your question..."
                className="w-full resize-none rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-foreground">
                Options (select the correct one)
              </label>
              <div className="space-y-2">
                {options.map((opt, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setCorrectAnswer(i)}
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 transition-colors ${
                        correctAnswer === i
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-input bg-background text-muted-foreground hover:border-primary/50"
                      }`}
                      aria-label={`Mark option ${i + 1} as correct`}
                    >
                      {correctAnswer === i && (
                        <CheckCircle2 className="h-4 w-4" />
                      )}
                    </button>
                    <input
                      required
                      value={opt}
                      onChange={(e) => updateOption(i, e.target.value)}
                      placeholder={`Option ${i + 1}`}
                      className="flex-1 rounded-lg border border-input bg-background px-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                    />
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between">
                <label htmlFor="timelimit" className="text-sm font-medium text-foreground">
                  Time Limit (seconds)
                </label>
                <span className="text-sm font-bold text-primary">
                  {timeLimit}s
                </span>
              </div>
              <input
                id="timelimit"
                type="range"
                min={10}
                max={120}
                step={5}
                value={timeLimit}
                onChange={(e) => setTimeLimit(parseInt(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          </div>

          <div className="mt-6 flex gap-3">
            <button
              type="submit"
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
            >
              <Plus className="h-4 w-4" />
              Add Question
            </button>
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="rounded-lg bg-secondary px-5 py-2.5 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
            >
              Cancel
            </button>
          </div>
        </motion.form>
      ) : (
        <div className="flex gap-3">
          <button
            onClick={() => setShowForm(true)}
            className="inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            Add Question
          </button>
          {questions.length > 0 && (
            <Link
              href={`/recruiter/jobs/${jobId}/candidates`}
              className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-5 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-secondary"
            >
              View Candidates
            </Link>
          )}
        </div>
      )}
    </PageTransition>
  )
}
