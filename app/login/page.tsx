"use client"

import { Suspense, useState, useEffect } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { motion } from "framer-motion"
import { Brain, Briefcase, GraduationCap, ArrowRight } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import { toast } from "sonner"

function LoginContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { login, signup, user } = useAuth()

  const preselectedRole = searchParams.get("role") as
    | "recruiter"
    | "student"
    | null

  const [mode, setMode] = useState<AuthMode>("signup")
  const [selectedRole, setSelectedRole] = useState<
    "recruiter" | "student" | null
  >(preselectedRole)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  useEffect(() => {
    if (user) {
      router.push(user.role === "recruiter" ? "/recruiter" : "/student")
    }
  }, [user, router])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (!selectedRole || !email || !password || (mode === "signup" && !name)) {
      toast.error("Please fill in all required fields")
      return
    }

    if (mode === "signup") {
      const result = signup({ name, email, password, role: selectedRole })
      if (!result.ok) {
        toast.error(result.error ?? "Signup failed")
        return
      }
      toast.success("Account created successfully")
      router.push(selectedRole === "recruiter" ? "/recruiter" : "/student")
      return
    }

    const result = login(email, password, selectedRole)
    if (!result.ok) {
      toast.error(result.error ?? "Login failed")
      return
    }
    toast.success("Welcome back!")
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
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-primary">
            <Brain className="h-8 w-8 text-primary-foreground" />
          </div>
          <h1 className="mb-2 text-2xl font-bold text-foreground">
            Welcome to HireSense AI
          </h1>
          <p className="text-muted-foreground">
            {mode === "signup" ? "Create an account" : "Login to your account"}
          </p>
        </div>

        <div className="mb-5 grid grid-cols-2 rounded-lg bg-muted p-1">
          <button
            type="button"
            onClick={() => setMode("signup")}
            className={`rounded-md px-3 py-1.5 text-sm font-medium ${
              mode === "signup" ? "bg-background text-foreground" : "text-muted-foreground"
            }`}
          >
            Sign Up
          </button>
          <button
            type="button"
            onClick={() => setMode("login")}
            className={`rounded-md px-3 py-1.5 text-sm font-medium ${
              mode === "login" ? "bg-background text-foreground" : "text-muted-foreground"
            }`}
          >
            Login
          </button>
        </div>

        <form onSubmit={handleSubmit}>
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
              <span className={`text-sm font-semibold ${selectedRole === "recruiter" ? "text-primary" : "text-foreground"}`}>
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
              <span className={`text-sm font-semibold ${selectedRole === "student" ? "text-primary" : "text-foreground"}`}>
                Student
              </span>
            </button>
          </div>

          <div className="space-y-4">
            {mode === "signup" && (
              <div>
                <label htmlFor="name" className="mb-1.5 block text-sm font-medium text-foreground">
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
            )}
            <div>
              <label htmlFor="email" className="mb-1.5 block text-sm font-medium text-foreground">
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
            <div>
              <label htmlFor="password" className="mb-1.5 block text-sm font-medium text-foreground">
                Password
              </label>
              <input
                id="password"
                type="password"
                minLength={6}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={!selectedRole || !email || !password || (mode === "signup" && !name)}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
          >
            {mode === "signup" ? "Create Account" : "Login"}
            <ArrowRight className="h-4 w-4" />
          </button>
        </form>
      </motion.div>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center">Loading...</div>}>
      <LoginContent />
    </Suspense>
  )
}
