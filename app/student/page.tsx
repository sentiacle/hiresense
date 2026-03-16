"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import {
  Briefcase,
  MapPin,
  Timer,
  CheckCircle2,
  ArrowRight,
} from "lucide-react"
import { getJobs, getApplication, getQuestionsForJob } from "@/lib/store"
import { useAuth } from "@/lib/auth-context"
import { PageTransition } from "@/components/shared/page-transition"
import useSWR from "swr"

export default function StudentJobsPage() {
  const { user } = useAuth()

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

  const jobs = data ?? []

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

                <p className="mb-4 line-clamp-3 text-sm text-muted-foreground leading-relaxed">
                  {job.description}
                </p>

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
    </PageTransition>
  )
}
