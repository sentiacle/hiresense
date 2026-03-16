# HireSense Project Understanding

## 1) Product Goal
HireSense is an AI-assisted recruitment platform that lets:
- **Recruiters** create jobs, define scoring weights, set pass thresholds, create timed MCQ questions, and review ranked candidates.
- **Students** browse jobs, submit CV text, complete timed quizzes, and receive pass/fail plus success probability feedback.

The app combines CV scoring + test grading + probability prediction into a single flow.

---

## 2) High-Level Architecture
The repository is split into three major runtime areas:

1. **Next.js frontend + API routes** (`app/`, `lib/`, `components/`)
   - UI, role-based flows, local persistence, and API endpoints.
   - Frontend can call `/api/*` endpoints.

2. **FastAPI backend** (`backend/`)
   - Provides CV analysis, entity extraction, test grading, and success prediction.
   - Loads a BERT + BiLSTM + CRF model if available; otherwise uses heuristics.

3. **Model training pipeline** (`scripts/training/`)
   - Training/evaluation/export scripts for the NER model used by backend.

---

## 3) Frontend Structure (Next.js App Router)

### Routing and role flows
- `/` landing page with value proposition and links to role-based login.
- `/login` pseudo-auth form (name/email/role).
- `/recruiter/*` recruiter dashboard and management pages.
- `/student/*` student browsing and application flow.

### Auth/session model
- Authentication is local-only via `AuthProvider` and `localStorage` key `hiresense_user`.
- On app load, seed data is initialized, then user is restored from local storage.

### Data persistence model
This is a demo-style state layer using browser local storage (`lib/store.ts`):
- Jobs: `hiresense_jobs`
- Questions: `hiresense_questions`
- Applications: `hiresense_applications`
- Initial seed marker: `hiresense_seeded`

CRUD helpers are provided for jobs/questions/applications and are consumed by pages via SWR fetchers.

### Recruiter UX flow
1. Create job with required skills, category weights, and minimum thresholds.
2. Add/edit/delete timed MCQ questions per job.
3. View candidates table with filters/sorting and expandable CV-score breakdown.

### Student UX flow
1. Browse job cards.
2. Open apply page:
   - Paste CV text.
   - Complete timed MCQ quiz (auto-submit when timer expires).
3. System computes:
   - CV score + category breakdown,
   - test score,
   - success probability,
   - pass/fail decision based on job thresholds.
4. Application is persisted and results view is shown.

---

## 4) Scoring Logic in Frontend (`lib/scoring.ts`)

### CV analysis modes
- `analyzeCvWithDL(...)` calls `/api/analyze-cv` and expects model-style response.
- If API fails/unavailable, it falls back to local heuristic `analyzeCv(...)`.

### Heuristic CV scoring categories
- Skills
- Experience
- Projects
- Achievements
- Education

Each category has pattern/keyword extraction logic, then weighted aggregation based on recruiter-configured weights.

### Test grading
- Percentage of correct answers.
- Returns 100 when there are zero questions.

### Success prediction
- Uses weighted combination of CV and test scores (60/40), then a sigmoid transform centered around 0.5.

---

## 5) Next API Routes (`app/api/*`)

- `POST /api/analyze-cv`
  - First attempts FastAPI backend (`PYTHON_BACKEND_URL`, default `http://localhost:8000`).
  - On failure, returns heuristic result in backend-compatible response shape.

- `POST /api/grade-test`
  - Delegates to `gradeTest` helper.

- `POST /api/predict-success`
  - Delegates to `predictSuccess` helper.

This dual-path design makes the app usable even when Python services are down.

---

## 6) Python Backend (`backend/`)

### Service shape
FastAPI app with CORS enabled and startup lifespan that initializes model manager.

Endpoints:
- `GET /health`
- `POST /analyze-cv`
- `POST /extract-entities`
- `POST /grade-test`
- `POST /predict-success`

### Model loading strategy
`ModelManager` attempts to load `MODEL_PATH` (default `./models/resume_ner_model.pt`):
- If found + valid, runs trained BERT+BiLSTM+CRF inference.
- If missing/failing, initializes tokenizer and falls back to heuristic entity extraction.

### Backend scoring engine
`backend/scoring.py` computes category scores from extracted entities + CV text:
- skill matching against required skills,
- experience years/seniority,
- degree/GPA/relevance,
- projects and delivery verbs,
- achievements/certifications/publications.

Returns weighted total score + detailed breakdown for API response.

---

## 7) Training Pipeline (`scripts/training/`)

The training folder contains end-to-end model development utilities:
- `train.py`: training loop, optimizer/scheduler setup, evaluation, checkpointing.
- `dataset.py`: dataset loading/token-label alignment and dataloaders.
- `model.py`: BERT + BiLSTM + CRF architecture.
- `evaluate.py`: model evaluation and metrics reporting.
- `export_model.py`: export trained artifact(s) for deployment.
- `config.py`: label maps and train/model/data config objects.

Together these scripts support producing the checkpoint consumed by backend `ModelManager`.

---

## 8) Key Design Characteristics

### Strengths
- Full recruiter/student product flow is implemented end-to-end.
- Works in degraded mode without Python model service.
- Reasonable separation between UI, store, scoring logic, and backend services.
- Includes model training/export path in same repo.

### Constraints / trade-offs
- Frontend persistence and auth are local-storage based (no multi-user server state).
- No hardened auth/authorization (demo-only identity model).
- Mixed scoring behavior can diverge between frontend heuristics and backend model.
- Production readiness would require DB, auth, tenant isolation, and observability.

---

## 9) How to Run (practical map)

### Frontend
- `npm run dev` (or equivalent package-manager command)
- Access Next app and use role-based flows.

### Backend
- Run FastAPI app in `backend/` (e.g., uvicorn).
- Optionally set `PYTHON_BACKEND_URL` for Next API integration.
- Optionally provide `MODEL_PATH` to trained checkpoint.

### Training
- Use scripts in `scripts/training/` with requirements from that folder.
- Export model checkpoint and place where backend can load it.

---

## 10) Bottom-line Understanding
HireSense is a full-stack hiring workflow prototype with:
- polished recruiter/student interfaces,
- deterministic local data model for demo usability,
- a layered CV analysis strategy (trained model path with heuristic fallback),
- and integrated ML training/export assets.

It is best characterized as a **production-leaning prototype**: feature-complete for demonstration and experimentation, with clear paths to harden for real-world deployment.
