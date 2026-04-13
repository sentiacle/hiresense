"""
HireSense AI — Model Architecture + Scoring
BERT + BiLSTM + CRF, Multi-Sector Resume NER

Key improvements for higher scoring:
  ─ CVScorer._extract_jd_terms: richer JD parsing — skills pulled from
    bullet points, commas, "+" separators, and inline mentions
  ─ CVScorer._match_score: uses shared SKILL_SYNONYMS from config so
    "pytorch" matches "torch", "gcp" matches "google cloud", etc.
  ─ CVScorer.score: project/achievement score no longer defaults to 1.0
    when JD has no requirements — uses a presence bonus instead
  ─ BertBiLSTMCRF: FP16-safe CRF (always float32), improved get_entities
"""

import re
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from torchcrf import CRF
from typing import Optional, Dict, List, Tuple

from config import (
    ModelConfig, NUM_LABELS, LABEL2ID, ID2LABEL,
    ENTITY_GROUPS, ScoringConfig, SKILL_SYNONYMS,
)


# =============================================================================
# Core model
# =============================================================================

class BertBiLSTMCRF(nn.Module):
    """
    BERT-base-uncased → BiLSTM → CRF

    CRF always runs in float32 even under AMP to prevent NaN gradients.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config     = config
        self.num_labels = NUM_LABELS

        self.bert = BertModel.from_pretrained(config.bert_model_name)
        if config.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=config.bert_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
        )
        self.dropout    = nn.Dropout(config.lstm_dropout)
        self.hidden2tag = nn.Linear(config.lstm_output_size, self.num_labels)
        self.crf        = CRF(self.num_labels, batch_first=True)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

    def _get_emissions(self, input_ids, attention_mask):
        bert_out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out    = bert_out.last_hidden_state
        lstm_out,_ = self.lstm(seq_out)
        lstm_out   = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out).float()   # always float32 for CRF

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask)
        mask      = attention_mask.bool()
        outputs   = {"emissions": emissions}

        if labels is not None:
            lbl_crf = labels.clone()
            lbl_crf[labels == -100] = 0
            loss = -self.crf(emissions, lbl_crf, mask=mask, reduction="mean")
            outputs["loss"] = loss

        with torch.no_grad():
            outputs["predictions"] = self.crf.decode(emissions, mask=mask)

        return outputs

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)
        return out["emissions"], out["predictions"]

    def get_entities(self, tokens, predictions, word_ids):
        """
        Map subword predictions → word-level BIO entities.
        Returns list of {"text", "label", "start", "end"}.
        """
        # Build word_idx → first prediction mapping
        word_preds = []
        seen       = set()
        ptr        = 0
        for wid in word_ids:
            if wid is None:
                ptr += 1
                continue
            if ptr >= len(predictions): break
            if wid not in seen:
                seen.add(wid)
                word_preds.append((wid, predictions[ptr]))
            ptr += 1

        entities = []
        current  = None

        for word_idx, pred_id in word_preds:
            if word_idx >= len(tokens): continue
            label = ID2LABEL.get(pred_id, "O")
            token = tokens[word_idx]

            if label.startswith("B-"):
                if current: entities.append(current)
                current = {"text": token, "label": label[2:],
                           "start": word_idx, "end": word_idx}
            elif label.startswith("I-") and current:
                if label[2:] == current["label"]:
                    current["text"] += " " + token
                    current["end"]   = word_idx
                else:
                    entities.append(current)
                    current = None
            else:
                if current: entities.append(current)
                current = None

        if current: entities.append(current)
        return entities


# =============================================================================
# Inference wrapper
# =============================================================================

class ModelForInference:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt        = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = ckpt.get("config", ModelConfig())
        self.model  = BertBiLSTMCRF(self.config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.bert_model_name)

    def extract_entities(self, text: str) -> List[Dict]:
        if not text or not text.strip(): return []
        tokens = re.findall(r"\b\w[\w'.+-]*\b|[^\w\s]", text)
        if not tokens: return []

        enc = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        word_ids = enc.word_ids()
        if not word_ids: return []

        _, preds = self.model.predict(ids, mask)
        if not preds or not preds[0]: return []
        return self.model.get_entities(tokens, preds[0], word_ids)


# =============================================================================
# CV Scorer — the core of the HireSense scoring pipeline
# =============================================================================

class CVScorer:
    """
    Scores a CV against a Job Description.

    Improvements over previous version:
      1. _extract_jd_terms parses bullet-point JDs, comma lists, and
         inline mentions — not just the section after a header keyword.
      2. _match_score uses the shared SKILL_SYNONYMS table so aliases
         (pytorch ↔ torch, gcp ↔ google cloud) always match.
      3. Project and achievement categories use a "presence bonus" when
         the JD has no explicit requirements (freshers benefit).
      4. Partial-word matching is length-gated to avoid false positives
         ("c" matching "react" etc.).
    """

    # Build a flat lookup: alias → canonical for fast reverse lookup
    _ALIAS_TO_CANONICAL: Dict[str, str] = {}
    for _canon, _aliases in SKILL_SYNONYMS.items():
        for _a in _aliases:
            _ALIAS_TO_CANONICAL[_a] = _canon

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _canonicalise(self, term: str) -> str:
        """Map any alias to its canonical form."""
        n = self._norm(term)
        return self._ALIAS_TO_CANONICAL.get(n, n)

    def _expand(self, terms: List[str]) -> List[str]:
        """Return canonicalised + all aliases for each term."""
        expanded = set()
        for t in terms:
            canon = self._canonicalise(t)
            expanded.add(canon)
            # Add all known aliases
            if canon in SKILL_SYNONYMS:
                for alias in SKILL_SYNONYMS[canon]:
                    expanded.add(self._norm(alias))
            # If the raw term itself is in the synonym table, add its aliases too
            n = self._norm(t)
            if n in SKILL_SYNONYMS:
                for alias in SKILL_SYNONYMS[n]:
                    expanded.add(self._norm(alias))
        return list(expanded)

    # ── JD term extraction ────────────────────────────────────────────────────

    def _extract_jd_terms(self, jd_text: str) -> Dict[str, List[str]]:
        """
        Extract required terms from a raw JD string.

        Strategy (applied in order, terms deduplicated per category):
          a) Skill-like tokens after section headers
          b) Comma/semicolon/bullet-delimited lists anywhere in the JD
          c) Quoted or parenthesised terms
          d) Education and experience patterns via regex
        """
        tl = jd_text.lower()

        # ── Skills ──────────────────────────────────────────────────────────
        skill_tokens: List[str] = []

        # After section headers
        m = re.search(
            r"(skills?|requirements?|qualifications?|technologies?|tools?|"
            r"tech stack|must have|nice.to.have)[:\s]+(.*?)(?=\n\n|\Z)",
            tl, re.DOTALL | re.IGNORECASE,
        )
        if m:
            raw = m.group(2)
        else:
            raw = tl

        # Split on common delimiters used in JDs
        for chunk in re.split(r"[,;•\-\|/\n]", raw):
            chunk = chunk.strip()
            # Accept multi-word chunks up to 4 words (e.g. "machine learning")
            if 1 < len(chunk) <= 40:
                # Further split on "and", "or", "+"
                for sub in re.split(r"\band\b|\bor\b|\+", chunk):
                    sub = sub.strip()
                    if 1 < len(sub) <= 30:
                        skill_tokens.append(self._norm(sub))

        # Also grab anything in parentheses or square brackets
        for m2 in re.finditer(r"[\(\[](.*?)[\)\]]", tl):
            for sub in re.split(r"[,;/]", m2.group(1)):
                sub = sub.strip()
                if 1 < len(sub) <= 25:
                    skill_tokens.append(self._norm(sub))

        # Remove obvious noise (stop words, very short non-alpha tokens)
        STOP = {"the","a","an","to","of","for","in","with","on","at","is","be",
                "and","or","we","you","our","your","will","can","may","should"}
        skill_tokens = [t for t in skill_tokens
                        if t not in STOP and len(t) > 1 and re.search(r"[a-z]", t)]

        # ── Education ────────────────────────────────────────────────────────
        edu_re = re.compile(
            r"(bachelor|master|phd|doctorate|mba|b\.?tech|m\.?tech|b\.?e|b\.?sc|"
            r"m\.?sc|llb|llm|ca|cfa|cpa|diploma|degree|b\.?s|m\.?s)[a-z ]{0,50}",
            re.IGNORECASE,
        )
        edu_terms = [self._norm(m.group(0)) for m in edu_re.finditer(tl)]

        # ── Experience ───────────────────────────────────────────────────────
        exp_re = re.compile(
            r"\d+[\+]?\s*(?:years?|yrs?)[a-z ,\-]{0,50}(?:experience|exp)?"
            r"|(?:senior|lead|manager|director|head|principal|junior|associate|"
            r"intern|fresher|graduate)[a-z ]{0,40}",
            re.IGNORECASE,
        )
        exp_terms = [self._norm(m.group(0)) for m in exp_re.finditer(tl)]

        return {
            "skill":       list(dict.fromkeys(skill_tokens)),  # dedup, preserve order
            "education":   list(dict.fromkeys(edu_terms)),
            "experience":  list(dict.fromkeys(exp_terms)),
            "project":     [],
            "achievement": [],
        }

    # ── Match scoring ─────────────────────────────────────────────────────────

    def _match_score(
        self,
        cv_terms: List[str],
        jd_terms: List[str],
        category: str,
    ) -> float:
        """
        Recall-oriented: what fraction of JD requirements does the CV cover?

        Special cases:
          • project / achievement with empty JD terms → presence bonus
            (how many entities the CV has, scaled against a baseline of 3)
          • All other empty JD terms → full credit (requirement not stated)
        """
        if not jd_terms:
            if category in ("project", "achievement"):
                # Freshers: reward having projects/achievements even if JD doesn't list them
                bonus = min(len(cv_terms) / 3.0, 1.0)
                return bonus
            return 1.0   # requirement not stated → full credit

        if not cv_terms:
            return 0.0

        jd_expanded = self._expand([self._norm(t) for t in jd_terms])
        cv_expanded = self._expand([self._norm(t) for t in cv_terms])

        matched = 0
        for jd_t in jd_expanded:
            if len(jd_t) < 2: continue
            for cv_t in cv_expanded:
                if len(cv_t) < 2: continue
                # Exact match or substring match (length-gated to avoid false positives)
                if jd_t == cv_t:
                    matched += 1; break
                if len(jd_t) >= 4 and len(cv_t) >= 4:
                    if jd_t in cv_t or cv_t in jd_t:
                        matched += 1; break

        return min(matched / max(len(jd_expanded), 1), 1.0)

    # ── Public scoring interface ──────────────────────────────────────────────

    def score(
        self,
        cv_entities: List[Dict],
        jd_text: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Score a CV against a job description.

        Args:
            cv_entities : list of {text, label, start, end} from NER model
            jd_text     : raw job description text
            weights     : per-category weights (will be normalised to sum=1)

        Returns:
            {
              "overall": float (0–100),
              "breakdown": {
                "skill":       {"score", "weight", "cv_count",
                                "jd_terms_count", "weighted_score"},
                ...
              },
              "matched_entities": cv_entities,
            }
        """
        if weights is None:
            weights = dict(self.config.default_weights)

        # Map frontend plural keys to backend singular categories
        singular_to_plural = {}
        mapped_weights = {}
        for k, v in weights.items():
            norm_k = k
            if k == "skills": norm_k = "skill"
            elif k == "projects": norm_k = "project"
            elif k == "achievements": norm_k = "achievement"
            mapped_weights[norm_k] = v
            singular_to_plural[norm_k] = k
        weights = mapped_weights

        # Normalise weights
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}

        # Group CV entities by scoring category
        cv_by_cat: Dict[str, List[str]] = {k: [] for k in weights}
        for ent in cv_entities:
            label = ent["label"].upper()
            for cat, group_labels in ENTITY_GROUPS.items():
                if cat not in cv_by_cat: continue
                # group_labels like ["B-SKILL","I-SKILL"] — strip prefix
                cat_label = group_labels[0].replace("B-", "")
                if label == cat_label:
                    cv_by_cat[cat].append(ent["text"])

        jd_by_cat = self._extract_jd_terms(jd_text)

        breakdown:    Dict   = {}
        weighted_sum: float  = 0.0

        for cat, weight in weights.items():
            cv_terms = cv_by_cat.get(cat, [])
            jd_terms = jd_by_cat.get(cat, [])
            raw      = self._match_score(cv_terms, jd_terms, cat)

            original_k = singular_to_plural.get(cat, cat)
            breakdown[original_k] = {
                "score":          round(raw * 100, 2),
                "weight":         round(weight, 4),
                "cv_count":       len(cv_terms),
                "jd_terms_count": len(jd_terms),
                "weighted_score": round(raw * weight * 100, 2),
            }
            weighted_sum += raw * weight

        return {
            "overall":          round(weighted_sum * 100, 2),
            "breakdown":        breakdown,
            "matched_entities": cv_entities,
        }


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    from config import ModelConfig, ScoringConfig

    cfg   = ModelConfig()
    model = BertBiLSTMCRF(cfg)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture : BERT + BiLSTM + CRF")
    print(f"Total params : {total:,}")
    print(f"Trainable    : {trainable:,}")
    print(f"Num labels   : {NUM_LABELS}")

    B, T = 2, 128
    ids  = torch.randint(0, 30000, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    lbls = torch.randint(0, NUM_LABELS, (B, T))
    out  = model(ids, mask, lbls)
    print(f"Forward pass : loss={out['loss'].item():.4f}  "
          f"emissions={out['emissions'].shape}  "
          f"preds={len(out['predictions'])} seqs\n")

    # Scorer test — simulates a Sanidhya Patel CV vs a Python backend JD
    scorer = CVScorer(ScoringConfig())
    entities = [
        {"text": "Python",       "label": "SKILL", "start": 0, "end": 0},
        {"text": "FastAPI",      "label": "SKILL", "start": 1, "end": 1},
        {"text": "Docker",       "label": "SKILL", "start": 2, "end": 2},
        {"text": "GCP",          "label": "SKILL", "start": 3, "end": 3},
        {"text": "PostgreSQL",   "label": "SKILL", "start": 4, "end": 4},
        {"text": "React",        "label": "SKILL", "start": 5, "end": 5},
        {"text": "Software Developer Intern", "label": "EXP",  "start": 6, "end": 8},
        {"text": "Founder Lead Engineer",     "label": "EXP",  "start": 9, "end": 11},
        {"text": "B.Tech Information Technology", "label": "EDU", "start": 12, "end": 14},
        {"text": "MERYT",        "label": "PROJ",  "start": 15, "end": 15},
        {"text": "BinSavvy",     "label": "PROJ",  "start": 16, "end": 16},
        {"text": "HackCrux 2025 Finalist", "label": "ACH", "start": 17, "end": 19},
        {"text": "Oracle Cloud Infrastructure Certified", "label": "CERT", "start":20, "end":23},
    ]
    jd = """
    Backend Engineer (Fresher / Intern)
    Required Skills: Python, FastAPI or Django, Docker, REST APIs, PostgreSQL or MySQL.
    Nice-to-have: GCP, AWS, React, Git, CI/CD.
    Education: B.Tech in Computer Science, IT, or related field.
    Experience: Internship or project experience in backend / AI development.
    """
    result = scorer.score(entities, jd)
    print(f"Overall score: {result['overall']:.1f} / 100")
    for cat, info in result["breakdown"].items():
        bar = "█" * int(info["score"] / 10)
        print(f"  {cat:12s}: {info['score']:5.1f}  w={info['weight']:.2f}  "
              f"cv={info['cv_count']}  jd_terms={info['jd_terms_count']}  {bar}")
