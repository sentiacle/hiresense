import { NextResponse } from "next/server"
import { analyzeCv } from "@/lib/scoring"
import type { WeightConfig } from "@/lib/types"

// Python backend URL (for Vercel Python Services)
const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL || "http://localhost:8000"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { cv_text, cvText, jd_text, jdText, required_skills, requiredSkills, weights } = body as {
      cv_text?: string
      cvText?: string
      jd_text?: string
      jdText?: string
      required_skills?: string[]
      requiredSkills?: string[]
      weights: WeightConfig
    }

    const text = cv_text || cvText
    const jd = jd_text || jdText || ""
    const skills = required_skills || requiredSkills || []

    if (!text || !weights) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      )
    }

    // Try to call Python backend first (BERT + BiLSTM + CRF model)
    try {
      const backendResponse = await fetch(`${PYTHON_BACKEND}/analyze-cv`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cv_text: text,
          jd_text: jd || undefined,   // FIX: forward JD so Python backend auto-extracts skills
          required_skills: skills,
          weights: {
            skills: weights.skills,
            experience: weights.experience,
            projects: weights.projects,
            achievements: weights.achievements,
            education: weights.education,
          },
        }),
      })

      if (backendResponse.ok) {
        const result = await backendResponse.json()
        return NextResponse.json(result)
      }
      
      console.log("[v0] Python backend returned non-OK, falling back to heuristic")
    } catch (backendError) {
      console.log("[v0] Python backend unavailable, using heuristic scoring:", backendError)
    }

    // Fallback to heuristic analysis
    const { score, breakdown } = analyzeCv(text, skills, weights)
    
    return NextResponse.json({
      success: true,
      entities: [],
      scores: {
        skills: { score: breakdown.skills, weight: weights.skills, entities: [], details: "Heuristic analysis" },
        experience: { score: breakdown.experience, weight: weights.experience, entities: [], details: "Heuristic analysis" },
        projects: { score: breakdown.projects, weight: weights.projects, entities: [], details: "Heuristic analysis" },
        achievements: { score: breakdown.achievements, weight: weights.achievements, entities: [], details: "Heuristic analysis" },
        education: { score: breakdown.education, weight: weights.education, entities: [], details: "Heuristic analysis" },
      },
      total_score: score,
      breakdown: {
        skills: [],
        experience: [],
        education: [],
        projects: [],
        achievements: [],
        organizations: [],
        dates: [],
      },
      model_info: {
        model_name: "heuristic",
        device: "cpu",
        model_type: "Fallback (Python backend unavailable)",
      },
    })
  } catch (error) {
    console.error("[v0] Error in analyze-cv route:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
