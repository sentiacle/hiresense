# HireSense

HireSense is an AI-driven recruitment platform with two roles:
- **Recruiter**: creates jobs, adds timed assessment questions, configures multifactor CV scoring weights, and sets pass criteria.
- **Student**: uploads CV text, completes timed questions, and sees CV score, test score, and probability of success.

## Deep Learning-first CV extraction

This project is designed so CV analysis is done using a **BERT + BiLSTM + CRF** NER pipeline when available.

Flow in production mode:
1. Frontend calls `POST /api/analyze-cv`.
2. Next API proxies request to FastAPI backend (`PYTHON_BACKEND_URL`).
3. FastAPI loads exported model checkpoint (`MODEL_PATH`) and extracts entities.
4. Scoring engine computes weighted category scores and total score.

If model/backend is unavailable, the app gracefully falls back to heuristic scoring so the site still works.

## Project structure

- `app/` — Next.js App Router frontend + API routes
- `lib/` — scoring helpers, auth context, local storage data store
- `backend/` — FastAPI service for CV analysis/NER/scoring endpoints
- `scripts/training/` — model training, evaluation, and export scripts

## Run frontend

```bash
npm install
npm run dev
```

## Run backend

```bash
cd backend
# install dependencies with your preferred tool (uv/pip)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Optional env vars:
- `PYTHON_BACKEND_URL` (for Next API route) default: `http://localhost:8000`
- `MODEL_PATH` (for FastAPI model manager) default: `./models/resume_ner_model.pt`

## Train model with resume datasets

Inside `scripts/training`:

1. Place datasets in configured paths (see `config.py`):
   - `./data/resume_corpus`
   - `./data/ner_annotated_cvs`
   - `./data/kaggle_resume_pdf` (Kaggle: `hadikp/resume-data-pdf`)
2. Run training:

```bash
cd scripts/training
pip install -r requirements.txt
python train.py
```

3. Export best checkpoint for backend inference:

```bash
python export_model.py
```

4. Point backend `MODEL_PATH` to the exported `resume_ner_model.pt`.

## Validate model quality (recommended)

After training, run evaluation to confirm model quality before deployment:

```bash
cd scripts/training
python evaluate.py
```

Check `output/evaluation_results.json` and aim for:
- strong overall entity F1 (commonly `>= 0.80` on your held-out set for production-like use)
- no critical category collapse (skills/experience/education should not be near zero F1)
- acceptable inference latency for your deployment target

## Notes

- Recruiter-configured weights (skills, experience, projects, achievements, education) are applied during score computation.
- Students receive CV score, test score, pass/fail status, and predicted hiring probability.
- Includes celebratory notifications for upload and completion milestones.
