import { NextResponse } from "next/server"
import { predictSuccess } from "@/lib/scoring"

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { cvScore, testScore } = body as {
      cvScore: number
      testScore: number
    }

    if (cvScore === undefined || testScore === undefined) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      )
    }

    const probability = predictSuccess(cvScore, testScore)
    return NextResponse.json({ probability })
  } catch {
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
