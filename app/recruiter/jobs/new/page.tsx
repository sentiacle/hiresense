"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import {
  ArrowLeft,
  Plus,
  X,
  AlertCircle,
  CheckCircle2,
} from "lucide-react"
import Link from "next/link"
import { createJob } from "@/lib/store"
import { useAuth } from "@/lib/auth-context"
import type { WeightConfig } from "@/lib/types"
import { toast } from "sonner"
import { PageTransition } from "@/components/shared/page-transition"

const WEIGHT_FACTORS: { key: keyof WeightConfig; label: string; description: string }[] = [
  { key: "skills", label: "Skills", description: "Technical and soft skills match" },
  { key: "experience", label: "Experience", description: "Years and relevance of experience" },
  { key: "projects", label: "Projects", description: "Personal and professional projects" },
  { key: "achievements", label: "Achievements", description: "Awards, certifications, publications" },
  { key: "education", label: "Education", description: "Degree level and relevance" },
]

export default function CreateJobPage() {
  const router = useRouter()
  const { user } = useAuth()

  const [title, setTitle] = useState("")
  const [company, setCompany] = useState("")
  const [description, setDescription] = useState("")
  const [skillInput, setSkillInput] = useState("")
  const [requiredSkills, setRequiredSkills] = useState<string[]>([])
  const [weights, setWeights] = useState<WeightConfig>({
    skills: 25,
    experience: 20,
    projects: 25,
    achievements: 15,
    education: 15,
  })
  const [minCvScore, setMinCvScore] = useState(50)
  const [minTestScore, setMinTestScore] = useState(50)

  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0)
  const isWeightValid = totalWeight === 100

  const addSkill = () => {
    const trimmed = skillInput.trim().toLowerCase()
    if (trimmed && !requiredSkills.includes(trimmed)) {
      setRequiredSkills([...requiredSkills, trimmed])
      setSkillInput("")
    }
  }

  const removeSkill = (skill: string) => {
    setRequiredSkills(requiredSkills.filter((s) => s !== skill))
  }

  const handleWeightChange = (key: keyof WeightConfig, value: number) => {
    setWeights((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!user || !isWeightValid) return

    const job = {
      id: `job-${Date.now()}`,
      title,
      company,
      description,
      requiredSkills,
      weights,
      minCvScore,
      minTestScore,
      recruiterId: user.id,
      createdAt: new Date().toISOString(),
    }

    createJob(job)
    toast.success("Job created successfully! Now add some questions.")
    router.push(`/recruiter/jobs/${job.id}/questions`)
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
        <h1 className="text-3xl font-bold text-foreground">Create New Job</h1>
        <p className="mt-1 text-muted-foreground">
          Set up a job posting with custom scoring weights and passing criteria.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="max-w-3xl space-y-8">
        {/* Basic Info */}
        <div className="rounded-xl border border-border/60 bg-card p-6">
          <h2 className="mb-4 text-lg font-semibold text-foreground">
            Job Details
          </h2>
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="title" className="mb-1.5 block text-sm font-medium text-foreground">
                  Job Title
                </label>
                <input
                  id="title"
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g., Frontend Engineer"
                  className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                />
              </div>
              <div>
                <label htmlFor="company" className="mb-1.5 block text-sm font-medium text-foreground">
                  Company Name
                </label>
                <input
                  id="company"
                  required
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="e.g., TechCorp Inc."
                  className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                />
              </div>
            </div>
            <div>
              <label htmlFor="description" className="mb-1.5 block text-sm font-medium text-foreground">
                Job Description
              </label>
              <textarea
                id="description"
                required
                rows={5}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe the role, responsibilities, and what you're looking for..."
                className="w-full resize-none rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </div>

            {/* Skills */}
            <div>
              <label className="mb-1.5 block text-sm font-medium text-foreground">
                Required Skills
              </label>
              <div className="flex gap-2">
                <input
                  value={skillInput}
                  onChange={(e) => setSkillInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault()
                      addSkill()
                    }
                  }}
                  placeholder="Type a skill and press Enter"
                  className="flex-1 rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                />
                <button
                  type="button"
                  onClick={addSkill}
                  className="inline-flex items-center gap-1 rounded-lg bg-secondary px-4 py-2.5 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
                >
                  <Plus className="h-4 w-4" />
                  Add
                </button>
              </div>
              {requiredSkills.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {requiredSkills.map((skill) => (
                    <span
                      key={skill}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-primary/10 px-3 py-1 text-sm font-medium text-primary"
                    >
                      {skill}
                      <button
                        type="button"
                        onClick={() => removeSkill(skill)}
                        className="rounded-full hover:bg-primary/20"
                      >
                        <X className="h-3.5 w-3.5" />
                        <span className="sr-only">Remove {skill}</span>
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Weight Configuration */}
        <div className="rounded-xl border border-border/60 bg-card p-6">
          <div className="mb-1 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground">
              Scoring Weights
            </h2>
            <div
              className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium ${
                isWeightValid
                  ? "bg-success/10 text-success"
                  : "bg-destructive/10 text-destructive"
              }`}
            >
              {isWeightValid ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : (
                <AlertCircle className="h-3.5 w-3.5" />
              )}
              Total: {totalWeight}%
            </div>
          </div>
          <p className="mb-6 text-sm text-muted-foreground">
            Adjust how much each factor contributes to the CV score. Must total
            100%.
          </p>

          <div className="space-y-5">
            {WEIGHT_FACTORS.map((factor) => (
              <div key={factor.key}>
                <div className="mb-2 flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-foreground">
                      {factor.label}
                    </span>
                    <span className="ml-2 text-xs text-muted-foreground">
                      {factor.description}
                    </span>
                  </div>
                  <span className="min-w-[3rem] text-right text-sm font-bold text-primary">
                    {weights[factor.key]}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={weights[factor.key]}
                  onChange={(e) =>
                    handleWeightChange(factor.key, parseInt(e.target.value))
                  }
                  className="w-full accent-primary"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Passing Criteria */}
        <div className="rounded-xl border border-border/60 bg-card p-6">
          <h2 className="mb-1 text-lg font-semibold text-foreground">
            Passing Criteria
          </h2>
          <p className="mb-6 text-sm text-muted-foreground">
            Set minimum scores required for a candidate to pass.
          </p>
          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label htmlFor="minCv" className="text-sm font-medium text-foreground">
                  Min CV Score
                </label>
                <span className="text-sm font-bold text-primary">
                  {minCvScore}/100
                </span>
              </div>
              <input
                id="minCv"
                type="range"
                min={0}
                max={100}
                step={5}
                value={minCvScore}
                onChange={(e) => setMinCvScore(parseInt(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label htmlFor="minTest" className="text-sm font-medium text-foreground">
                  Min Test Score
                </label>
                <span className="text-sm font-bold text-primary">
                  {minTestScore}/100
                </span>
              </div>
              <input
                id="minTest"
                type="range"
                min={0}
                max={100}
                step={5}
                value={minTestScore}
                onChange={(e) => setMinTestScore(parseInt(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          </div>
        </div>

        {/* Submit */}
        <motion.button
          type="submit"
          disabled={!title || !description || !isWeightValid}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          className="w-full rounded-xl bg-primary px-6 py-3.5 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
        >
          Create Job & Add Questions
        </motion.button>
      </form>
    </PageTransition>
  )
}
