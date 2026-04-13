# HireSense AI — Deployment Package

## Files

| File | Purpose |
|------|---------|
| `resume_ner_model.pt` | Full PyTorch checkpoint (model + config + labels) |
| `model_weights.pt` | Weights-only file (smaller) |
| `model_config.json` | Architecture + label map |
| `tokenizer/` | BERT tokenizer files |
| `inference.py` | Backend inference helper |
| `scorer.py` | CV-vs-JD scoring helper |

## Quick Start

```python
from inference import ResumeNERPipeline
from scorer import score_cv

pipeline = ResumeNERPipeline(model_dir="./models", device="cpu")

# Extract entities from a CV
entities = pipeline.extract(cv_text)

# Score against a JD with recruiter-configured weights
result = score_cv(
    cv_entities=entities,
    jd_text=jd_text,
    weights={
        "skill":       0.35,
        "experience":  0.30,
        "education":   0.15,
        "project":     0.10,
        "achievement": 0.10,
    }
)

print(result["overall"])     # e.g. 78.4  (0–100)
print(result["breakdown"])   # per-category scores
```

## Entity Types

| Label | Description | Examples |
|-------|-------------|---------|
| SKILL | Skills & tools (all sectors) | Python, Tally, AutoCAD, Litigation |
| EXP | Job titles / experience | Senior Engineer, Accountant |
| EDU | Education & qualifications | B.Tech, CA, MBBS, LLB |
| PROJ | Projects | Built recommendation engine |
| ACH | Awards, publications, patents | Best Auditor Award 2022 |
| CERT | Certifications | AWS Certified, PMP |
| ORG | Employers, universities | Google, IIT Bombay |
| LOC | Locations | Bangalore, Mumbai |
| DATE | Dates & durations | 2018–Present |
| NAME | Candidate name | Priya Sharma |
| CONTACT | Email, phone | priya@email.com |
| SECTOR | Industry sector tag | Finance, Healthcare |

## Sectors Supported

Tech · Finance · Accounting · HR · Sales · Marketing · Design ·
Architecture · Civil Engineering · Mechanical Engineering ·
Electrical Engineering · Healthcare · Legal · Agriculture ·
Education · Aviation · BPO · Operations · Blockchain · Data Science
