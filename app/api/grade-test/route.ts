import { NextResponse } from "next/server"
import { gradeTest } from "@/lib/scoring"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { answers, correctAnswers } = body as {
      answers: number[]
      correctAnswers: number[]
    }

    if (!answers || !correctAnswers) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      )
    }

    const score = gradeTest(answers, correctAnswers)
    return NextResponse.json({ score })
  } catch {
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
