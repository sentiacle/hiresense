"use client"

import { useState } from "react"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import {
  Briefcase,
  MapPin,
  Timer,
  CheckCircle2,
  ArrowRight,
  X,
  ChevronRight,
} from "lucide-react"
import { getJobs, getApplication, getQuestionsForJob } from "@/lib/store"
import { useAuth } from "@/lib/auth-context"
import { PageTransition } from "@/components/shared/page-transition"
import useSWR from "swr"

type JobWithMeta = {
  id: string
  title: string
  company: string
  description: string
  requiredSkills: string[]
  applied: boolean
  questionCount: number
  [key: string]: unknown
}

export default function StudentJobsPage() {
  const { user } = useAuth()
  const [selectedJob, setSelectedJob] = useState<JobWithMeta | null>(null)

  const { data } = useSWR(
    user ? `student-jobs-${user.id}` : null,
    () => {
      const jobs = getJobs()
      return jobs.map((job) => ({
        ...job,
        applied: user ? !!getApplication(user.id, job.id) : false,
        questionCount: getQuestionsForJob(job.id).length,
      }))
    }
  )

  const jobs: JobWithMeta[] = data ?? []

  return (
    <PageTransition>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground">Available Jobs</h1>
        <p className="mt-1 text-muted-foreground">
          Browse open positions and apply with your CV. You'll take a timed
          assessment and receive instant AI-powered feedback.
        </p>
      </div>

      {jobs.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border py-16">
          <Briefcase className="mb-4 h-12 w-12 text-muted-foreground/40" />
          <p className="mb-1 text-lg font-medium text-muted-foreground">
            No jobs available
          </p>
          <p className="text-sm text-muted-foreground">
            Check back soon for new openings!
          </p>
        </div>
      ) : (
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {jobs.map((job, i) => (
            <motion.div
              key={job.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: i * 0.06 }}
              className="group flex flex-col rounded-xl border border-border/60 bg-card transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="flex-1 p-6">
                <div className="mb-3 flex items-start justify-between">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
                    <Briefcase className="h-5 w-5 text-primary" />
                  </div>
                  {job.applied && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                      <CheckCircle2 className="h-3 w-3" />
                      Applied
                    </span>
                  )}
                </div>

                <h3 className="mb-1 text-lg font-semibold text-foreground">
                  {job.title}
                </h3>
                <p className="mb-3 flex items-center gap-1 text-sm text-muted-foreground">
                  <MapPin className="h-3.5 w-3.5" />
                  {job.company}
                </p>

                <p className="mb-2 line-clamp-3 text-sm text-muted-foreground leading-relaxed">
                  {job.description}
                </p>

                {/* View full description link */}
                <button
                  onClick={() => setSelectedJob(job)}
                  className="mb-4 flex items-center gap-0.5 text-xs font-medium text-primary hover:underline"
                >
                  View full description
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>

                <div className="mb-4 flex flex-wrap gap-1.5">
                  {job.requiredSkills.slice(0, 5).map((skill) => (
                    <span
                      key={skill}
                      className="rounded-md bg-primary/8 px-2 py-0.5 text-xs font-medium text-primary"
                    >
                      {skill}
                    </span>
                  ))}
                  {job.requiredSkills.length > 5 && (
                    <span className="rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                      +{job.requiredSkills.length - 5}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Timer className="h-3.5 w-3.5" />
                    {job.questionCount} questions
                  </span>
                </div>
              </div>

              <div className="border-t border-border/40 p-4">
                {job.applied ? (
                  <Link
                    href={`/student/apply/${job.id}`}
                    className="flex w-full items-center justify-center gap-2 rounded-lg bg-secondary px-4 py-2.5 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
                  >
                    View Results
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                ) : (
                  <Link
                    href={`/student/apply/${job.id}`}
                    className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                  >
                    Apply Now
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Job Detail Modal */}
      <AnimatePresence>
        {selectedJob && (
          <>
            {/* Backdrop */}
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
              onClick={() => setSelectedJob(null)}
            />

            {/* Modal panel */}
            <motion.div
              key="modal"
              initial={{ opacity: 0, scale: 0.96, y: 16 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 16 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className="fixed inset-x-4 top-1/2 z-50 mx-auto max-w-2xl -translate-y-1/2 overflow-hidden rounded-2xl border border-border/60 bg-card shadow-2xl"
            >
              {/* Modal header */}
              <div className="flex items-start justify-between border-b border-border/40 p-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-primary/10">
                    <Briefcase className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-foreground">
                      {selectedJob.title}
                    </h2>
                    <p className="flex items-center gap-1 text-sm text-muted-foreground">
                      <MapPin className="h-3.5 w-3.5" />
                      {selectedJob.company}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedJob(null)}
                  className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {/* Modal body — scrollable */}
              <div className="max-h-[60vh] overflow-y-auto p-6">
                {/* Status badge */}
                {selectedJob.applied && (
                  <span className="mb-4 inline-flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-semibold text-success">
                    <CheckCircle2 className="h-3 w-3" />
                    Already Applied
                  </span>
                )}

                {/* Full description */}
                <div className="mb-6">
                  <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                    Job Description
                  </h3>
                  <p className="text-sm leading-relaxed text-foreground whitespace-pre-line">
                    {selectedJob.description}
                  </p>
                </div>

                {/* Skills */}
                <div className="mb-6">
                  <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                    Required Skills
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedJob.requiredSkills.map((skill) => (
                      <span
                        key={skill}
                        className="rounded-md bg-primary/8 px-2.5 py-1 text-xs font-medium text-primary"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Assessment info */}
                <div className="rounded-lg border border-border/50 bg-muted/40 p-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Timer className="h-4 w-4 shrink-0 text-primary" />
                    <span>
                      This role includes a timed assessment of{" "}
                      <span className="font-semibold text-foreground">
                        {selectedJob.questionCount} question
                        {selectedJob.questionCount !== 1 ? "s" : ""}
                      </span>
                      . You'll receive instant AI-powered feedback after
                      submission.
                    </span>
                  </div>
                </div>
              </div>

              {/* Modal footer */}
              <div className="border-t border-border/40 p-4">
                {selectedJob.applied ? (
                  <Link
                    href={`/student/apply/${selectedJob.id}`}
                    onClick={() => setSelectedJob(null)}
                    className="flex w-full items-center justify-center gap-2 rounded-lg bg-secondary px-4 py-2.5 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
                  >
                    View Results
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                ) : (
                  <Link
                    href={`/student/apply/${selectedJob.id}`}
                    onClick={() => setSelectedJob(null)}
                    className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                  >
                    Apply Now
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </PageTransition>
  )
}
