"use client"

import { useState, useMemo } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import {
  ArrowLeft,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Users,
  CheckCircle2,
  XCircle,
  Filter,
  Download,
} from "lucide-react"
import { getJob, getApplicationsForJob, getQuestionsForJob } from "@/lib/store"
import { PageTransition } from "@/components/shared/page-transition"
import useSWR from "swr"

type SortKey = "studentName" | "cvScore" | "testScore" | "probability"
type FilterStatus = "all" | "passed" | "failed"

export default function CandidatesPage() {
  const params = useParams()
  const jobId = params.jobId as string

  const { data } = useSWR(`candidates-${jobId}`, () => ({
    job: getJob(jobId),
    applications: getApplicationsForJob(jobId),
    questions: getQuestionsForJob(jobId),
  }))

  const job = data?.job
  const applications = data?.applications ?? []
  const questions = data?.questions ?? []

  const [sortKey, setSortKey] = useState<SortKey>("probability")
  const [sortAsc, setSortAsc] = useState(false)
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all")
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const filteredAndSorted = useMemo(() => {
    let filtered = applications
    if (filterStatus === "passed") {
      filtered = applications.filter((a) => a.passed)
    } else if (filterStatus === "failed") {
      filtered = applications.filter((a) => !a.passed)
    }

    return [...filtered].sort((a, b) => {
      const aVal = a[sortKey]
      const bVal = b[sortKey]
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return sortAsc
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number)
    })
  }, [applications, sortKey, sortAsc, filterStatus])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(false)
    }
  }

  if (!job) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-muted-foreground">Job not found.</p>
      </div>
    )
  }

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortKey !== columnKey)
      return <ArrowUpDown className="h-3.5 w-3.5 text-muted-foreground/50" />
    return sortAsc ? (
      <ChevronUp className="h-3.5 w-3.5" />
    ) : (
      <ChevronDown className="h-3.5 w-3.5" />
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
          Candidates for {job.title}
        </h1>
        <p className="mt-1 text-muted-foreground">
          Review candidate performance and submitted CVs.
        </p>
      </div>

      <div className="mb-4 flex items-center gap-3">
        <Filter className="h-4 w-4 text-muted-foreground" />
        {(["all", "passed", "failed"] as FilterStatus[]).map((status) => (
          <button
            key={status}
            onClick={() => setFilterStatus(status)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              filterStatus === status
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        ))}
        <span className="ml-auto text-sm text-muted-foreground">
          {filteredAndSorted.length} candidate{filteredAndSorted.length !== 1 && "s"}
        </span>
      </div>

      {filteredAndSorted.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border py-16">
          <Users className="mb-4 h-12 w-12 text-muted-foreground/40" />
          <p className="mb-1 text-lg font-medium text-muted-foreground">
            No candidates yet
          </p>
          <p className="text-sm text-muted-foreground">
            {filterStatus !== "all"
              ? "No candidates match this filter."
              : "Candidates will appear here once they apply."}
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/60">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/40 bg-muted/50">
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  <button
                    onClick={() => handleSort("studentName")}
                    className="flex items-center gap-1"
                  >
                    Name <SortIcon columnKey="studentName" />
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  <button
                    onClick={() => handleSort("cvScore")}
                    className="flex items-center gap-1"
                  >
                    CV Score <SortIcon columnKey="cvScore" />
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  <button
                    onClick={() => handleSort("testScore")}
                    className="flex items-center gap-1"
                  >
                    Test Score <SortIcon columnKey="testScore" />
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  <button
                    onClick={() => handleSort("probability")}
                    className="flex items-center gap-1"
                  >
                    Probability <SortIcon columnKey="probability" />
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  CV
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredAndSorted.map((app, i) => (
                <motion.tr
                  key={app.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.03 }}
                  className="group cursor-pointer border-b border-border/30 last:border-0 hover:bg-muted/30"
                  onClick={() =>
                    setExpandedId(expandedId === app.id ? null : app.id)
                  }
                >
                  <td className="px-4 py-3">
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        {app.studentName}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {app.studentEmail}
                      </p>
                    </div>
                  </td>
                  <td className="px-4 py-3">{app.cvScore}</td>
                  <td className="px-4 py-3">{app.testScore}</td>
                  <td className="px-4 py-3">
                    <span className="text-sm font-bold text-primary">
                      {app.probability}%
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    {app.cvFileDataUrl ? (
                      <a
                        href={app.cvFileDataUrl}
                        download={app.cvFileName || `${app.studentName}-cv.pdf`}
                        onClick={(e) => e.stopPropagation()}
                        className="inline-flex items-center gap-1 rounded-md bg-secondary px-2 py-1 text-xs font-medium text-secondary-foreground"
                      >
                        <Download className="h-3 w-3" /> Download
                      </a>
                    ) : (
                      <span className="text-xs text-muted-foreground">Unavailable</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {app.passed ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                        <CheckCircle2 className="h-3 w-3" />
                        Passed
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-semibold text-destructive">
                        <XCircle className="h-3 w-3" />
                        Failed
                      </span>
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>

          <AnimatePresence>
            {expandedId && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden border-t border-border/40 bg-muted/20"
              >
                {(() => {
                  const app = filteredAndSorted.find((a) => a.id === expandedId)
                  if (!app) return null

                  return (
                    <div className="space-y-4 px-6 py-4">
                      <h4 className="text-sm font-semibold text-foreground">
                        CV Score Breakdown for {app.studentName}
                      </h4>

                      <div className="grid gap-3 sm:grid-cols-5">
                        {[
                          { label: "Skills", value: app.breakdown.skills },
                          { label: "Experience", value: app.breakdown.experience },
                          { label: "Projects", value: app.breakdown.projects },
                          { label: "Achievements", value: app.breakdown.achievements },
                          { label: "Education", value: app.breakdown.education },
                        ].map((f) => (
                          <div key={f.label}>
                            <p className="mb-1 text-xs text-muted-foreground">{f.label}</p>
                            <p className="text-sm font-semibold text-foreground">{f.value}</p>
                          </div>
                        ))}
                      </div>

                      <div>
                        <h5 className="mb-2 text-sm font-semibold text-foreground">
                          Question-by-question answers
                        </h5>
                        <div className="space-y-2">
                          {questions.map((q, idx) => {
                            const selected = app.answers[idx]
                            const selectedText =
                              selected === -1 || selected === undefined
                                ? "No answer"
                                : q.options[selected] ?? "Invalid answer"
                            const correctText = q.options[q.correctAnswer] ?? "N/A"
                            return (
                              <div
                                key={q.id}
                                className="rounded-lg border border-border/50 bg-background p-3"
                              >
                                <p className="text-sm font-medium text-foreground">
                                  Q{idx + 1}. {q.question}
                                </p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Student answer: <span className="font-medium">{selectedText}</span>
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  Correct answer: <span className="font-medium">{correctText}</span>
                                </p>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </PageTransition>
  )
}
