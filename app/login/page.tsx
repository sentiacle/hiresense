"use client"

import { useState, useEffect } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { motion } from "framer-motion"
import { Brain, Briefcase, GraduationCap, ArrowRight } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import { toast } from "sonner"

export default function LoginPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { login, user } = useAuth()

  const preselectedRole = searchParams.get("role") as
    | "recruiter"
    | "student"
    | null
  const [selectedRole, setSelectedRole] = useState<
    "recruiter" | "student" | null
  >(preselectedRole)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")

  useEffect(() => {
    if (user) {
      router.push(user.role === "recruiter" ? "/recruiter" : "/student")
    }
  }, [user, router])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedRole || !name || !email) return

    login({
      id: `${selectedRole}-${Date.now()}`,
      name,
      email,
      role: selectedRole,
    })

    toast.success(
      selectedRole === "recruiter"
        ? "Welcome aboard, Recruiter! Let's find great talent!"
        : "Welcome! Let's find your dream job!"
    )
    router.push(selectedRole === "recruiter" ? "/recruiter" : "/student")
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-background via-background to-primary/5 px-4">
      <div className="pointer-events-none absolute -top-40 right-20 h-80 w-80 rounded-full bg-primary/8 blur-3xl" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-primary">
            <Brain className="h-8 w-8 text-primary-foreground" />
          </div>
          <h1 className="mb-2 text-2xl font-bold text-foreground">
            Welcome to HireSense AI
          </h1>
          <p className="text-muted-foreground">
            Choose your role and sign in to get started.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Role Selection */}
          <div className="mb-6 grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setSelectedRole("recruiter")}
              className={`flex flex-col items-center gap-2 rounded-xl border-2 p-5 transition-all ${
                selectedRole === "recruiter"
                  ? "border-primary bg-primary/5 shadow-md shadow-primary/10"
                  : "border-border bg-card hover:border-primary/30"
              }`}
            >
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-xl transition-colors ${
                  selectedRole === "recruiter"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                <Briefcase className="h-6 w-6" />
              </div>
              <span
                className={`text-sm font-semibold ${
                  selectedRole === "recruiter"
                    ? "text-primary"
                    : "text-foreground"
                }`}
              >
                Recruiter
              </span>
            </button>

            <button
              type="button"
              onClick={() => setSelectedRole("student")}
              className={`flex flex-col items-center gap-2 rounded-xl border-2 p-5 transition-all ${
                selectedRole === "student"
                  ? "border-primary bg-primary/5 shadow-md shadow-primary/10"
                  : "border-border bg-card hover:border-primary/30"
              }`}
            >
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-xl transition-colors ${
                  selectedRole === "student"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                <GraduationCap className="h-6 w-6" />
              </div>
              <span
                className={`text-sm font-semibold ${
                  selectedRole === "student"
                    ? "text-primary"
                    : "text-foreground"
                }`}
              >
                Student
              </span>
            </button>
          </div>

          {/* Form fields */}
          <div className="space-y-4">
            <div>
              <label
                htmlFor="name"
                className="mb-1.5 block text-sm font-medium text-foreground"
              >
                Full Name
              </label>
              <input
                id="name"
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter your full name"
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </div>
            <div>
              <label
                htmlFor="email"
                className="mb-1.5 block text-sm font-medium text-foreground"
              >
                Email Address
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={!selectedRole || !name || !email}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
          >
            Continue as{" "}
            {selectedRole
              ? selectedRole.charAt(0).toUpperCase() + selectedRole.slice(1)
              : "..."}
            <ArrowRight className="h-4 w-4" />
          </button>
        </form>
      </motion.div>
    </div>
  )
}
