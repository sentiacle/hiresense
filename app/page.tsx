"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import {
  Brain,
  FileText,
  Timer,
  BarChart3,
  Sparkles,
  ArrowRight,
  CheckCircle2,
} from "lucide-react"

const features = [
  {
    icon: Brain,
    title: "AI-Powered CV Analysis",
    description:
      "Our intelligent system extracts skills, experience, projects, and achievements from CVs to generate comprehensive scores.",
  },
  {
    icon: Timer,
    title: "Timed Assessments",
    description:
      "Create custom MCQ tests with per-question time limits. Auto-graded with instant results for candidates.",
  },
  {
    icon: BarChart3,
    title: "Multi-Factor Scoring",
    description:
      "Configure custom weights for skills, experience, projects, achievements, and education. Your criteria, your way.",
  },
  {
    icon: Sparkles,
    title: "Smart Matching",
    description:
      "Our prediction engine combines CV and test scores to estimate each candidate's probability of success.",
  },
]

const steps = [
  "Recruiter posts a job with custom scoring weights",
  "Students upload CVs and take timed assessments",
  "AI analyzes CVs and auto-grades tests",
  "Candidates ranked with pass/fail based on your criteria",
]

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Nav */}
      <header className="fixed top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Brain className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold text-foreground">
              HireSense AI
            </span>
          </Link>
          <Link
            href="/login"
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Get Started
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative flex min-h-[90vh] items-center justify-center overflow-hidden pt-16">
        {/* Gradient background */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-primary/10" />
        <div className="pointer-events-none absolute -top-40 right-0 h-96 w-96 rounded-full bg-primary/10 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-40 left-0 h-96 w-96 rounded-full bg-accent/10 blur-3xl" />

        <div className="relative mx-auto max-w-4xl px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 text-sm font-medium text-primary">
              <Sparkles className="h-4 w-4" />
              AI-Powered Recruitment Platform
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mb-6 text-balance text-5xl font-bold leading-tight tracking-tight text-foreground md:text-6xl lg:text-7xl"
          >
            Smarter Hiring,{" "}
            <span className="bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
              Better Matches
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mx-auto mb-10 max-w-2xl text-pretty text-lg text-muted-foreground md:text-xl"
          >
            Screen resumes with AI, assess candidates with timed tests, and find
            the perfect match -- all in one platform. Configure your own scoring
            criteria and let our system do the heavy lifting.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col items-center justify-center gap-4 sm:flex-row"
          >
            <Link
              href="/login?role=recruiter"
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-8 py-3.5 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:bg-primary/90 hover:shadow-xl hover:shadow-primary/30 sm:w-auto"
            >
              <FileText className="h-5 w-5" />
              I'm a Recruiter
            </Link>
            <Link
              href="/login?role=student"
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl border-2 border-primary/20 bg-background px-8 py-3.5 text-base font-semibold text-foreground transition-all hover:border-primary/40 hover:bg-primary/5 sm:w-auto"
            >
              <Brain className="h-5 w-5" />
              I'm a Student
            </Link>
          </motion.div>
        </div>
      </section>

      {/* How it works */}
      <section className="border-t border-border/50 bg-muted/30 py-24">
        <div className="mx-auto max-w-6xl px-6">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-16 text-center"
          >
            <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
              How It Works
            </h2>
            <p className="mx-auto max-w-xl text-muted-foreground">
              A simple, streamlined process from job posting to candidate
              ranking.
            </p>
          </motion.div>
          <div className="mx-auto max-w-2xl">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
                className="flex items-start gap-4 pb-8 last:pb-0"
              >
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  {i + 1}
                </div>
                <div className="flex items-center gap-3 pt-1">
                  <CheckCircle2 className="h-5 w-5 shrink-0 text-primary/60" />
                  <p className="text-foreground">{step}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24">
        <div className="mx-auto max-w-6xl px-6">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-16 text-center"
          >
            <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
              Powerful Features
            </h2>
            <p className="mx-auto max-w-xl text-muted-foreground">
              Everything you need to make data-driven hiring decisions.
            </p>
          </motion.div>
          <div className="grid gap-6 md:grid-cols-2">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
                className="group rounded-2xl border border-border/60 bg-card p-8 transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 text-primary transition-colors group-hover:bg-primary group-hover:text-primary-foreground">
                  <feature.icon className="h-6 w-6" />
                </div>
                <h3 className="mb-2 text-xl font-semibold text-foreground">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-border/50 bg-primary/5 py-24">
        <div className="mx-auto max-w-3xl px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
              Ready to Transform Your Hiring?
            </h2>
            <p className="mb-8 text-lg text-muted-foreground">
              Join HireSense AI today and experience the future of recruitment.
            </p>
            <Link
              href="/login"
              className="inline-flex items-center gap-2 rounded-xl bg-primary px-8 py-3.5 text-base font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:bg-primary/90 hover:shadow-xl"
            >
              Get Started Now
              <ArrowRight className="h-5 w-5" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 bg-background py-8">
        <div className="mx-auto max-w-6xl px-6 text-center">
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Brain className="h-4 w-4" />
            <span>HireSense AI -- Intelligent Recruitment Platform</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
