"use client"

import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react"
import type { User } from "./types"
import { seedDataIfNeeded } from "./store"

interface RegisteredUser extends User {
  password: string
}

interface AuthContextType {
  user: User | null
  login: (
    email: string,
    password: string,
    role: "recruiter" | "student"
  ) => { ok: boolean; error?: string }
  signup: (payload: {
    name: string
    email: string
    password: string
    role: "recruiter" | "student"
  }) => { ok: boolean; error?: string }
  logout: () => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  login: () => ({ ok: false, error: "Not initialized" }),
  signup: () => ({ ok: false, error: "Not initialized" }),
  logout: () => {},
  isLoading: true,
})

const AUTH_KEY = "hiresense_user"
const USERS_KEY = "hiresense_users"

function getUsers(): RegisteredUser[] {
  try {
    const raw = localStorage.getItem(USERS_KEY)
    return raw ? (JSON.parse(raw) as RegisteredUser[]) : []
  } catch {
    return []
  }
}

function setUsers(users: RegisteredUser[]) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users))
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    seedDataIfNeeded()
    try {
      const stored = localStorage.getItem(AUTH_KEY)
      if (stored) {
        setUser(JSON.parse(stored))
      }
    } catch {
      // ignore
    }
    setIsLoading(false)
  }, [])

  const login = (
    email: string,
    password: string,
    role: "recruiter" | "student"
  ) => {
    const users = getUsers()
    const match = users.find(
      (u) =>
        u.email.toLowerCase() === email.toLowerCase() &&
        u.password === password &&
        u.role === role
    )

    if (!match) {
      return { ok: false, error: "Invalid credentials for selected role" }
    }

    const safeUser: User = {
      id: match.id,
      name: match.name,
      email: match.email,
      role: match.role,
    }

    setUser(safeUser)
    localStorage.setItem(AUTH_KEY, JSON.stringify(safeUser))
    return { ok: true }
  }

  const signup = ({
    name,
    email,
    password,
    role,
  }: {
    name: string
    email: string
    password: string
    role: "recruiter" | "student"
  }) => {
    const users = getUsers()
    const exists = users.some((u) => u.email.toLowerCase() === email.toLowerCase())
    if (exists) return { ok: false, error: "Email already exists" }

    const newUser: RegisteredUser = {
      id: `${role}-${Date.now()}`,
      name,
      email,
      role,
      password,
    }

    users.push(newUser)
    setUsers(users)

    const safeUser: User = {
      id: newUser.id,
      name: newUser.name,
      email: newUser.email,
      role: newUser.role,
    }

    setUser(safeUser)
    localStorage.setItem(AUTH_KEY, JSON.stringify(safeUser))
    return { ok: true }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem(AUTH_KEY)
  }

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
