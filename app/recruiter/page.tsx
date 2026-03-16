"use client"

import { useMemo } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import {
  Briefcase,
  Users,
  FileText,
  TrendingUp,
  PlusCircle,
  ArrowRight,
  Timer,
  CheckCircle2,
  Trash2,
} from "lucide-react"
import { PageTransition } from "@/components/shared/page-transition"
import {
  getJobs,
  getApplications,
  getQuestionsForJob,
  deleteJob,
} from "@/lib/store"
import useSWR from "swr"
import { toast } from "sonner"

function useRecruiterData() {
  return useSWR("recruiter-dashboard", () => {
    const jobs = getJobs()
    const applications = getApplications()
    return { jobs, applications }
  })
}

export default function RecruiterDashboard() {
  const { data, mutate } = useRecruiterData()
  const jobs = data?.jobs ?? []
  const applications = data?.applications ?? []

  const stats = useMemo(() => {
    const avgCv =
      applications.length > 0
        ? Math.round(
            applications.reduce((s, a) => s + a.cvScore, 0) /
              applications.length
          )
        : 0
    const avgTest =
      applications.length > 0
        ? Math.round(
            applications.reduce((s, a) => s + a.testScore, 0) /
              applications.length
          )
        : 0
    return {
      totalJobs: jobs.length,
      totalCandidates: applications.length,
      avgCvScore: avgCv,
      avgTestScore: avgTest,
    }
  }, [jobs, applications])

  const statCards = [
    {
      label: "Active Jobs",
      value: stats.totalJobs,
      icon: Briefcase,
      color: "text-primary",
      bg: "bg-primary/10",
    },
    {
      label: "Total Candidates",
      value: stats.totalCandidates,
      icon: Users,
      color: "text-chart-4",
      bg: "bg-chart-4/10",
    },
    {
      label: "Avg CV Score",
      value: stats.avgCvScore,
      icon: FileText,
      color: "text-chart-3",
      bg: "bg-chart-3/10",
    },
    {
      label: "Avg Test Score",
      value: stats.avgTestScore,
      icon: TrendingUp,
      color: "text-chart-2",
      bg: "bg-chart-2/10",
    },
  ]

  return (
    <PageTransition>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="mt-1 text-muted-foreground">
            Overview of your recruitment activity.
          </p>
        </div>
        <Link
          href="/recruiter/jobs/new"
          className="inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90"
        >
          <PlusCircle className="h-4 w-4" />
          Create Job
        </Link>
      </div>

      <div className="mb-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: i * 0.08 }}
            className="rounded-xl border border-border/60 bg-card p-6"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">
                {stat.label}
              </span>
              <div
                className={`flex h-10 w-10 items-center justify-center rounded-lg ${stat.bg}`}
              >
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
              </div>
            </div>
            <p className="mt-2 text-3xl font-bold text-foreground">
              {stat.value}
            </p>
          </motion.div>
        ))}
      </div>

      <div>
        <h2 className="mb-4 text-xl font-semibold text-foreground">Your Jobs</h2>
        {jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border py-16">
            <Briefcase className="mb-4 h-12 w-12 text-muted-foreground/40" />
            <p className="mb-2 text-lg font-medium text-muted-foreground">
              No jobs yet
            </p>
            <p className="mb-6 text-sm text-muted-foreground">
              Create your first job to start receiving applications.
            </p>
            <Link
              href="/recruiter/jobs/new"
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2 text-sm font-medium text-primary-foreground"
            >
              <PlusCircle className="h-4 w-4" />
              Create Job
            </Link>
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            {jobs.map((job, i) => {
              const jobApps = applications.filter((a) => a.jobId === job.id)
              const questionCount = getQuestionsForJob(job.id).length
              return (
                <motion.div
                  key={job.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.05 }}
                  className="group rounded-xl border border-border/60 bg-card p-6 transition-all hover:border-primary/30 hover:shadow-md"
                >
                  <div className="mb-3 flex items-start justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-foreground">
                        {job.title}
                      </h3>
                      <p className="text-sm text-muted-foreground">{job.company}</p>
                    </div>
                    <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                      Active
                    </span>
                  </div>

                  <p className="mb-4 line-clamp-2 text-sm text-muted-foreground leading-relaxed">
                    {job.description}
                  </p>

                  <div className="mb-4 flex flex-wrap gap-1.5">
                    {job.requiredSkills.slice(0, 4).map((skill) => (
                      <span
                        key={skill}
                        className="rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
                      >
                        {skill}
                      </span>
                    ))}
                    {job.requiredSkills.length > 4 && (
                      <span className="rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
                        +{job.requiredSkills.length - 4}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-4 border-t border-border/40 pt-4 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Users className="h-3.5 w-3.5" />
                      {jobApps.length} candidates
                    </span>
                    <span className="flex items-center gap-1">
                      <Timer className="h-3.5 w-3.5" />
                      {questionCount} questions
                    </span>
                    <span className="flex items-center gap-1">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Min CV: {job.minCvScore} | Test: {job.minTestScore}
                    </span>
                  </div>

                  <div className="mt-4 flex gap-2">
                    <Link
                      href={`/recruiter/jobs/${job.id}/questions`}
                      className="inline-flex items-center gap-1 rounded-lg bg-secondary px-3 py-1.5 text-xs font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
                    >
                      Questions
                    </Link>
                    <Link
                      href={`/recruiter/jobs/${job.id}/candidates`}
                      className="inline-flex items-center gap-1 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                      View Candidates
                      <ArrowRight className="h-3 w-3" />
                    </Link>
                    <button
                      onClick={() => {
                        if (!confirm("Delete this job? This also removes questions and applications.")) return
                        deleteJob(job.id)
                        mutate()
                        toast.success("Job deleted")
                      }}
                      className="inline-flex items-center gap-1 rounded-lg bg-destructive px-3 py-1.5 text-xs font-medium text-destructive-foreground transition-colors hover:opacity-90"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Delete
                    </button>
                  </div>
                </motion.div>
              )
            })}
          </div>
        )}
      </div>
    </PageTransition>
  )
}
