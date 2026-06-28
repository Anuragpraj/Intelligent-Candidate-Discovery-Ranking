#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker
================================================
Architecture: Weighted multi-component scoring with behavioral signal modifier.
NO network calls, NO GPU, NO LLM inference during ranking.
Designed to run in < 5 minutes on CPU with <= 16 GB RAM for 100K candidates.

Scoring Components:
  1. Title & Career Fit        (0.28) — production ML/AI background at product companies
  2. Core Skills Match         (0.25) — required skills with trust-weighted scoring
  3. Experience Band           (0.15) — 5-9 year band, quality-adjusted
  4. Location & Availability   (0.12) — India-preferenced, relocation willing
  5. Education Signal          (0.05) — tier bonus, not decisive
  6. Behavioral Signal         (0.15) — availability, engagement, responsiveness

Disqualifiers (applied before scoring):
  - Honeypot detection (impossible profiles)
  - Pure consulting background (TCS/Infosys/Wipro only career)
  - Non-technical titles (marketing, ops, HR as primary career)
  - CV/NLP/Speech-only background without NLP/IR
  - Recent-only AI (<12 months) with no prior ML production
"""

import json
import math
import csv
import sys
import argparse
from datetime import date, datetime
from pathlib import Path

# ── JD constants ──────────────────────────────────────────────────────────────

REQUIRED_SKILLS = {
    # Production retrieval/embedding
    "sentence-transformers", "sentence transformers", "embeddings", "embedding",
    "vector search", "vector database", "vector db",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "hybrid search", "dense retrieval", "semantic search",
    # Ranking / IR
    "ranking", "information retrieval", "bm25", "learning to rank", "reranking",
    "re-ranking", "ndcg", "mrr", "map", "retrieval", "search",
    # Core ML
    "python", "pytorch", "tensorflow", "transformers", "huggingface",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "nlp", "natural language processing", "llm", "large language model",
    "rag", "retrieval augmented generation",
    # Evaluation
    "a/b testing", "ab testing", "evaluation", "offline evaluation", "online evaluation",
    # Nice to have (lower weight)
    "xgboost", "lightgbm", "recommendation", "recommender", "distributed systems",
    "open source", "open-source",
}

NICE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "lightgbm", "recommendation", "recommender",
    "distributed systems", "open source", "open-source",
    "hr tech", "hrtech", "marketplace",
}

# Title signals — positive and negative
GOOD_TITLE_TOKENS = {
    "ml", "machine learning", "ai", "artificial intelligence",
    "nlp", "search", "ranking", "retrieval", "data scientist",
    "research engineer", "applied scientist", "applied ml",
    "applied ai", "senior engineer", "staff engineer",
    "tech lead", "engineering lead",
}
BAD_TITLE_TOKENS = {
    "marketing", "operations", "hr", "human resources", "content writer",
    "sales", "finance", "accounting", "mechanical", "civil", "electrical",
    "customer support", "customer success", "product manager",
    "scrum master", "business analyst",
}

# Companies that are pure consulting (disqualifier if ALL career is there)
PURE_CONSULTING = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "capgemini", "cognizant", "hcl", "tech mahindra", "mphasis",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "gurugram", "gurgaon", "india",
}

TODAY = date.today()

# ── Helper functions ──────────────────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def days_since(d_str):
    d = parse_date(d_str)
    if not d:
        return 999
    return (TODAY - d).days

def tokenize(text):
    """Lowercase, return set of words and common 2-grams."""
    if not text:
        return set()
    text = text.lower()
    words = text.replace("-", " ").replace("/", " ").split()
    tokens = set(words)
    # 2-grams
    for i in range(len(words) - 1):
        tokens.add(words[i] + " " + words[i+1])
    # 3-grams
    for i in range(len(words) - 2):
        tokens.add(words[i] + " " + words[i+1] + " " + words[i+2])
    return tokens

def skill_text_tokens(c):
    """All skill names lowercased."""
    tokens = set()
    for s in c.get("skills", []):
        tokens.update(tokenize(s.get("name", "")))
    return tokens

def career_text_tokens(c):
    """Tokens from all career descriptions and titles."""
    tokens = set()
    for job in c.get("career_history", []):
        tokens.update(tokenize(job.get("title", "")))
        tokens.update(tokenize(job.get("description", "")))
    tokens.update(tokenize(c.get("profile", {}).get("headline", "")))
    tokens.update(tokenize(c.get("profile", {}).get("summary", "")))
    return tokens

def all_tokens(c):
    return skill_text_tokens(c) | career_text_tokens(c)

# ── Honeypot detection ────────────────────────────────────────────────────────

def is_honeypot(c):
    """
    Detect subtly impossible profiles.
    Returns True if the candidate is likely a honeypot.
    """
    # Check 1: experience at company that's younger than claimed duration
    # (Hard to verify without company founding dates, but we can check duration_months vs years_of_experience)
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0
    career = c.get("career_history", [])

    # Check 2: expert skills with 0 duration and 0 endorsements across ALL skills
    skills = c.get("skills", [])
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 5:
        zero_duration_experts = [
            s for s in expert_skills
            if s.get("duration_months", 1) == 0 and s.get("endorsements", 1) == 0
        ]
        if len(zero_duration_experts) >= 4:
            return True

    # Check 3: Total duration_months in career >> years_of_experience * 12 by a lot
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 0 and total_career_months > (yoe * 12 + 36):  # allow 3yr overlap
        # But only flag if the discrepancy is huge (e.g. 200 months career but 5yr exp)
        if total_career_months > (yoe * 12) * 1.8:
            return True

    # Check 4: All skills listed as "expert" but profile_completeness very low (<40)
    pc = c.get("redrob_signals", {}).get("profile_completeness_score", 100)
    if len(expert_skills) >= 8 and pc < 40:
        return True

    return False

# ── Disqualifiers ─────────────────────────────────────────────────────────────

def get_disqualifier(c):
    """
    Returns a disqualifier string or None.
    Disqualified candidates get a heavy penalty but are not excluded
    (so they can still appear at ranks 90-100 if needed for the top-100 list).
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    tokens = all_tokens(c)

    # Check: entire career at pure consulting
    if career:
        companies = [j.get("company", "").lower() for j in career]
        is_consulting = all(
            any(cons in co for cons in PURE_CONSULTING)
            for co in companies
        )
        if is_consulting:
            return "pure_consulting"

    # Check: non-technical primary title
    current_title = profile.get("current_title", "").lower()
    if any(bad in current_title for bad in BAD_TITLE_TOKENS):
        return "non_technical_title"

    # Check: CV/speech/robotics without NLP/IR
    cv_tokens = {"computer vision", "opencv", "object detection", "image segmentation",
                 "speech recognition", "robotics", "ros", "lidar"}
    nlp_tokens = {"nlp", "natural language", "retrieval", "search", "ranking", "embeddings"}
    has_cv_only = cv_tokens & tokens and not (nlp_tokens & tokens)
    if has_cv_only:
        return "cv_only"

    return None

# ── Component scorers ─────────────────────────────────────────────────────────

def score_title_career(c):
    """
    0-1. Rewards ML/AI product-company career history.
    Penalizes consulting-heavy, non-AI backgrounds.
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    tokens = all_tokens(c)

    # Title relevance
    title = profile.get("current_title", "").lower()
    title_score = 0.0
    for tok in GOOD_TITLE_TOKENS:
        if tok in title:
            title_score = 1.0
            break
    if title_score == 0:
        # Check past titles
        for job in career:
            jtitle = job.get("title", "").lower()
            for tok in GOOD_TITLE_TOKENS:
                if tok in jtitle:
                    title_score = 0.6
                    break
            if title_score > 0:
                break

    # Production deployment signal from descriptions
    production_tokens = {
        "production", "deployed", "shipped", "real users", "at scale", "end-to-end",
        "retrieval", "ranking", "recommendation", "search", "embedding",
        "vector", "rag", "fine-tun",
    }
    prod_hits = sum(1 for pt in production_tokens if pt in " ".join(
        [j.get("description","") for j in career]).lower())
    production_score = min(1.0, prod_hits / 5.0)

    # Product company vs. consulting penalty
    product_months = 0
    consulting_months = 0
    for job in career:
        co = job.get("company", "").lower()
        months = job.get("duration_months", 0)
        if any(cons in co for cons in PURE_CONSULTING):
            consulting_months += months
        else:
            # Check industry
            ind = job.get("industry", "").lower()
            if ind not in ("it services", "consulting", "staffing"):
                product_months += months
            else:
                product_months += months * 0.5  # partial credit for IT services at product

    total_months = product_months + consulting_months + 1
    product_ratio = product_months / total_months

    score = 0.4 * title_score + 0.35 * production_score + 0.25 * product_ratio
    return min(1.0, score)

def score_skills(c):
    """
    0-1. Trust-weighted skill match.
    Weight = proficiency_multiplier × log(1 + endorsements) × log(1 + duration_months)
    then normalized by required skill hits.
    """
    skills = c.get("skills", [])
    sig = c.get("redrob_signals", {})
    assess = sig.get("skill_assessment_scores", {}) or {}

    proficiency_map = {"beginner": 0.25, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    required_hits = 0
    weighted_sum = 0.0
    nice_bonus = 0.0

    for sk in skills:
        name = sk.get("name", "").lower().strip()
        name_tokens = tokenize(name)
        prof = proficiency_map.get(sk.get("proficiency", "intermediate"), 0.5)
        end = sk.get("endorsements", 0) or 0
        dur = sk.get("duration_months", 1) or 1

        # Assessment score bonus
        assess_score = 0
        for ak, av in assess.items():
            if ak.lower() in name or name in ak.lower():
                assess_score = av / 100.0
                break

        # Trust multiplier
        trust = (
            prof *
            math.log(1 + end * 0.1 + 1) *
            math.log(1 + dur * 0.05 + 1) *
            (1 + 0.2 * assess_score)
        )

        # Check against required skills
        is_required = bool(name_tokens & REQUIRED_SKILLS or
                          any(req in name for req in REQUIRED_SKILLS))
        is_nice = bool(name_tokens & NICE_SKILLS or
                      any(req in name for req in NICE_SKILLS))

        if is_required:
            required_hits += 1
            weighted_sum += trust
        elif is_nice:
            nice_bonus += trust * 0.3

    # Normalize: 5 required skills with decent trust = score ~0.8
    base = min(1.0, weighted_sum / 8.0)
    bonus = min(0.15, nice_bonus / 10.0)

    # Career-text skill confirmation (weak signal)
    career_tokens = career_text_tokens(c)
    career_hits = len(career_tokens & REQUIRED_SKILLS)
    career_bonus = min(0.1, career_hits / 20.0)

    return min(1.0, base + bonus + career_bonus)

def score_experience(c):
    """
    0-1. Rewards 5-9 years. Penalizes <3 and >15.
    Also checks if experience is AI/ML-focused.
    """
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0

    # Band scoring
    if 5 <= yoe <= 9:
        band_score = 1.0
    elif 4 <= yoe < 5:
        band_score = 0.8
    elif 9 < yoe <= 12:
        band_score = 0.85
    elif 3 <= yoe < 4:
        band_score = 0.55
    elif 12 < yoe <= 15:
        band_score = 0.7
    elif yoe > 15:
        band_score = 0.5
    elif yoe < 3:
        band_score = 0.3
    else:
        band_score = 0.2

    # Quality multiplier: was the experience in relevant ML roles?
    career = c.get("career_history", [])
    ml_months = 0
    for job in career:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        if any(tok in title for tok in GOOD_TITLE_TOKENS) or \
           any(pt in desc for pt in ["embedding", "retrieval", "ranking", "search", "nlp", "llm"]):
            ml_months += job.get("duration_months", 0)

    total_months = yoe * 12 + 1
    ml_ratio = min(1.0, ml_months / total_months)
    quality_mult = 0.6 + 0.4 * ml_ratio

    return band_score * quality_mult

def score_location(c):
    """
    0-1. India + preferred cities get full score.
    Relocation willing gets partial. Outside India with no relocation: low.
    """
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})

    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    relocate = sig.get("willing_to_relocate", False)

    if country == "india":
        if any(city in location for city in PREFERRED_LOCATIONS):
            return 1.0
        elif relocate:
            return 0.85
        else:
            return 0.7
    else:
        if relocate:
            return 0.4
        return 0.1

def score_education(c):
    """
    0-1. Tier 1 & 2 institutions get bonus. CS/ML fields preferred.
    Deliberately low weight — not decisive.
    """
    edu = c.get("education", [])
    if not edu:
        return 0.4  # neutral, not punished

    best = 0.0
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.55,
                "tier_4": 0.35, "unknown": 0.45}
    relevant_fields = {"computer science", "cs", "ai", "machine learning",
                       "data science", "information technology", "it",
                       "electronics", "electrical", "statistics", "mathematics"}

    for e in edu:
        tier = tier_map.get(e.get("tier", "unknown"), 0.45)
        field = e.get("field_of_study", "").lower()
        field_bonus = 0.1 if any(f in field for f in relevant_fields) else 0
        score = min(1.0, tier + field_bonus)
        best = max(best, score)

    return best

def score_behavioral(c):
    """
    0-1. Availability and engagement signals from redrob_signals.
    This is a multiplier component — low behavioral = real candidate not reachable.
    """
    sig = c.get("redrob_signals", {})

    # Availability
    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.4
    days_inactive = days_since(sig.get("last_active_date"))
    if days_inactive < 7:
        recency = 1.0
    elif days_inactive < 30:
        recency = 0.85
    elif days_inactive < 90:
        recency = 0.6
    elif days_inactive < 180:
        recency = 0.35
    else:
        recency = 0.1

    # Responsiveness
    response_rate = sig.get("recruiter_response_rate", 0.5) or 0.5
    avg_resp_hrs = sig.get("avg_response_time_hours", 24) or 24
    resp_time_score = max(0, 1.0 - math.log(1 + avg_resp_hrs / 24) / 3.0)

    # Notice period (prefer <30 days)
    notice = sig.get("notice_period_days", 90) or 90
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.3

    # Engagement (profile completeness, verification)
    completeness = (sig.get("profile_completeness_score", 50) or 50) / 100.0
    verified = (
        (1 if sig.get("verified_email", False) else 0) +
        (1 if sig.get("verified_phone", False) else 0) +
        (1 if sig.get("linkedin_connected", False) else 0)
    ) / 3.0

    # GitHub activity (bonus for AI engineers)
    gh = sig.get("github_activity_score", -1)
    if gh == -1:
        gh_score = 0.4  # no GitHub, neutral
    else:
        gh_score = min(1.0, gh / 70.0)

    # Recruiter interest signals
    saved_30d = min(1.0, (sig.get("saved_by_recruiters_30d", 0) or 0) / 10.0)
    interview_rate = sig.get("interview_completion_rate", 0.5) or 0.5

    score = (
        0.20 * open_to_work +
        0.25 * recency +
        0.15 * response_rate +
        0.10 * resp_time_score +
        0.10 * notice_score +
        0.05 * completeness +
        0.05 * verified +
        0.05 * gh_score +
        0.03 * saved_30d +
        0.02 * interview_rate
    )
    return min(1.0, score)

# ── Main scorer ───────────────────────────────────────────────────────────────

WEIGHTS = {
    "title_career": 0.28,
    "skills":       0.25,
    "experience":   0.15,
    "behavioral":   0.15,
    "location":     0.12,
    "education":    0.05,
}

def score_candidate(c):
    components = {
        "title_career": score_title_career(c),
        "skills":       score_skills(c),
        "experience":   score_experience(c),
        "behavioral":   score_behavioral(c),
        "location":     score_location(c),
        "education":    score_education(c),
    }
    total = sum(WEIGHTS[k] * v for k, v in components.items())
    return total, components

def build_reasoning(c, components, rank, is_disq, disq_reason):
    """Generate a specific, honest 1-2 sentence reasoning string."""
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    career = c.get("career_history", [])

    name = profile.get("anonymized_name", "Candidate")
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "?")
    location = profile.get("location", "?")
    country = profile.get("country", "")

    # Top skills
    skills = c.get("skills", [])
    top_skills = [s["name"] for s in sorted(
        skills, key=lambda x: (x.get("endorsements",0), x.get("duration_months",0)), reverse=True
    )[:3]]
    top_skills_str = ", ".join(top_skills) if top_skills else "none listed"

    # Behavioral highlights
    response_rate = sig.get("recruiter_response_rate", 0)
    days_inactive = days_since(sig.get("last_active_date"))
    notice = sig.get("notice_period_days", 90)
    open_to_work = sig.get("open_to_work_flag", False)
    gh = sig.get("github_activity_score", -1)

    loc_str = f"{location}, {country}".strip(", ")

    if is_disq:
        return (
            f"{title} with {yoe:.1f} yrs; ranked low due to {disq_reason.replace('_',' ')} "
            f"and limited ML production signals. Skills: {top_skills_str}."
        )

    # Build contextual reasoning based on rank
    career_highlight = ""
    for job in career[:2]:
        desc = job.get("description", "")
        if any(kw in desc.lower() for kw in ["embedding", "retrieval", "ranking", "vector", "nlp", "llm", "rag"]):
            career_highlight = f"career includes {job.get('title','')} at {job.get('company','')} with production ML work"
            break

    parts = []
    # Opening: title + years + location
    parts.append(f"{title} with {yoe:.1f} yrs, based in {loc_str}")

    # Skill signal
    skill_score = components["skills"]
    if skill_score > 0.7:
        parts.append(f"strong skills match (top: {top_skills_str})")
    elif skill_score > 0.4:
        parts.append(f"partial skills match ({top_skills_str})")
    else:
        parts.append(f"limited core skills alignment")

    # Career signal
    if career_highlight:
        parts.append(career_highlight)
    elif components["title_career"] > 0.7:
        parts.append("strong AI/ML production career trajectory")
    elif components["title_career"] < 0.4:
        parts.append("limited ML production experience")

    # Behavioral concern
    concerns = []
    if days_inactive > 90:
        concerns.append(f"inactive {days_inactive}d")
    if response_rate < 0.3:
        concerns.append(f"low response rate ({response_rate:.0%})")
    if notice > 60:
        concerns.append(f"long notice ({notice}d)")
    if not open_to_work:
        concerns.append("not marked open-to-work")

    sentence1 = "; ".join(parts[:2]) + "."

    if concerns and rank > 20:
        sentence2 = f"Concerns: {', '.join(concerns)}."
    elif career_highlight and len(parts) > 2:
        sentence2 = parts[2].capitalize() + "."
    elif gh > 60:
        sentence2 = f"Active GitHub (score {gh}), good signal for open-source AI work."
    elif open_to_work and days_inactive < 14:
        sentence2 = "Actively open to work and recently engaged on platform."
    else:
        sentence2 = ""

    return (sentence1 + " " + sentence2).strip()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates.")

    print("Scoring candidates...")
    scored = []
    honeypot_count = 0
    for c in candidates:
        cid = c.get("candidate_id", "")

        # Honeypot detection
        if is_honeypot(c):
            honeypot_count += 1
            scored.append((cid, -0.5, {}, True, "honeypot"))
            continue

        # Disqualifier check
        disq = get_disqualifier(c)

        total, components = score_candidate(c)

        # Apply disqualifier penalty (don't eliminate — just heavily penalize)
        if disq:
            total = total * 0.25

        scored.append((cid, total, components, bool(disq), disq or ""))

    print(f"Honeypots detected: {honeypot_count}")

    # Sort by score descending, then candidate_id ascending for tie-break
    scored.sort(key=lambda x: (-x[1], x[0]))  # ascending candidate_id for tie-break

    # Take top N
    top = scored[:args.top]

    print(f"Writing top {args.top} to {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        # Build score → candidate map for reasoning
        cand_map = {c.get("candidate_id"): c for c in candidates}

        prev_score = None
        for rank_idx, (cid, score, components, is_disq, disq_reason) in enumerate(top):
            rank = rank_idx + 1

            # Ensure monotonically non-increasing scores
            if prev_score is not None and score > prev_score:
                score = prev_score
            prev_score = score

            c = cand_map.get(cid, {})
            reasoning = build_reasoning(c, components, rank, is_disq, disq_reason)

            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"Done. Submission written to {args.out}")
    print(f"Top 5 candidates: {[t[0] for t in top[:5]]}")

if __name__ == "__main__":
    main()
