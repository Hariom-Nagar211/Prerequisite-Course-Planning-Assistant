"""
evaluate.py — 25-query evaluation harness for MIT CS Catalog RAG Assistant

Runs all 25 test queries, scores responses, and writes a detailed report.

Run: python src/evaluate.py
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.rag_chain

from dotenv import load_dotenv
load_dotenv()

# ── 25-Query Test Set ──────────────────────────────────────────────────────
# Category codes:
#   PC  = Prerequisite Check (eligible / not eligible)    — 10 queries
#   PCC = Prerequisite Chain (multi-hop, 2+ steps)        —  5 queries
#   PR  = Program Requirement                             —  5 queries
#   ND  = Not-in-docs / trick question (abstention)       —  5 queries

TEST_SET = [
    # ── Category PC: Prerequisite Checks ──────────────────────────────────
    {
        "id": "PC-01",
        "category": "Prerequisite Check",
        "query": (
            "I have completed 6.1010 and 6.1200 (grade B in both). "
            "Can I enroll in 6.1210 next semester?"
        ),
        "expected_decision": "ELIGIBLE",
        "expected_keywords": ["6.1010", "6.1200", "C or better", "eligible"],
        "abstain_expected": False,
    },
    {
        "id": "PC-02",
        "category": "Prerequisite Check",
        "query": (
            "I have only completed 6.1010 (grade A). "
            "Can I enroll in 6.1210?"
        ),
        "expected_decision": "NOT ELIGIBLE",
        "expected_keywords": ["6.1200", "not eligible", "prerequisite"],
        "abstain_expected": False,
    },
    {
        "id": "PC-03",
        "category": "Prerequisite Check",
        "query": (
            "I completed 6.1200 and 6.1020 (both with B). "
            "Can I take 6.1800 (Computer Systems Engineering)?"
        ),
        "expected_decision": "ELIGIBLE",
        "expected_keywords": ["6.1020", "6.1200", "eligible", "6.1800"],
        "abstain_expected": False,
    },
    {
        "id": "PC-04",
        "category": "Prerequisite Check",
        "query": (
            "I have completed 6.1210 and 18.06 (Linear Algebra). "
            "Am I eligible for 6.3900 (Introduction to Machine Learning)?"
        ),
        "expected_decision": "ELIGIBLE",
        "expected_keywords": ["6.1210", "18.06", "eligible", "6.3900"],
        "abstain_expected": False,
    },
    {
        "id": "PC-05",
        "category": "Prerequisite Check",
        "query": (
            "I have only completed 6.1210 but NOT 18.06 or 18.C06. "
            "Can I take 6.3900?"
        ),
        "expected_decision": "NOT ELIGIBLE",
        "expected_keywords": ["18.06", "18.C06", "not eligible", "linear algebra"],
        "abstain_expected": False,
    },
    {
        "id": "PC-06",
        "category": "Prerequisite Check",
        "query": (
            "I completed 6.1020. Can I enroll in 6.1910 (Computation Structures)?"
        ),
        "expected_decision": "ELIGIBLE",
        "expected_keywords": ["6.1020", "eligible", "6.1910"],
        "abstain_expected": False,
    },
    {
        "id": "PC-07",
        "category": "Prerequisite Check",
        "query": (
            "I completed 6.1200 with a D. Can I use it as a prerequisite for 6.1210?"
        ),
        "expected_decision": "NOT ELIGIBLE",
        "expected_keywords": ["C or better", "D", "not eligible", "retake"],
        "abstain_expected": False,
    },
    {
        "id": "PC-08",
        "category": "Prerequisite Check",
        "query": (
            "I'm a freshman and have no CS courses yet. "
            "Can I enroll in 6.1010 (Fundamentals of Programming)?"
        ),
        "expected_decision": "ELIGIBLE",
        "expected_keywords": ["no prerequisite", "open", "eligible", "6.1010"],
        "abstain_expected": False,
    },
    {
        "id": "PC-09",
        "category": "Prerequisite Check",
        "query": (
            "I completed 6.1800 with an A. Can I take 6.5840 (Distributed Systems)?"
        ),
        "expected_decision": "ELIGIBLE",
        "expected_keywords": ["6.1800", "eligible", "6.5840"],
        "abstain_expected": False,
    },
    {
        "id": "PC-10",
        "category": "Prerequisite Check",
        "query": (
            "I completed 6.1020 but NOT 6.1210. Can I take 6.5060 (Algorithm Engineering)?"
        ),
        "expected_decision": "NOT ELIGIBLE",
        "expected_keywords": ["6.1210", "prerequisite", "not eligible"],
        "abstain_expected": False,
    },

    # ── Category PCC: Prerequisite Chain (multi-hop) ───────────────────────
    {
        "id": "PCC-01",
        "category": "Prerequisite Chain",
        "query": (
            "I just finished my first year and only completed 18.01 and 6.1010. "
            "What is the full chain of courses I need to eventually reach 6.3940 (Deep Learning)?"
        ),
        "expected_decision": "PATH",
        "expected_keywords": ["6.1200", "6.1210", "6.3900", "6.3940", "18.06"],
        "abstain_expected": False,
    },
    {
        "id": "PCC-02",
        "category": "Prerequisite Chain",
        "query": (
            "What courses must I complete, in order, to be eligible for "
            "6.4110 (Representation Learning)?"
        ),
        "expected_decision": "PATH",
        "expected_keywords": ["6.1210", "6.3900", "6.3940", "6.4110"],
        "abstain_expected": False,
    },
    {
        "id": "PCC-03",
        "category": "Prerequisite Chain",
        "query": (
            "I want to take 6.8610 (Quantitative NLP). "
            "What is the prerequisite chain starting from scratch?"
        ),
        "expected_decision": "PATH",
        "expected_keywords": ["6.3900", "6.3800", "6.8610", "6.1200", "6.1210"],
        "abstain_expected": False,
    },
    {
        "id": "PCC-04",
        "category": "Prerequisite Chain",
        "query": (
            "I completed 6.1010. What courses do I still need to take "
            "before I can enroll in 6.1800 (Computer Systems Engineering)?"
        ),
        "expected_decision": "PATH",
        "expected_keywords": ["6.1020", "6.1200", "6.1800"],
        "abstain_expected": False,
    },
    {
        "id": "PCC-05",
        "category": "Prerequisite Chain",
        "query": (
            "I completed only 18.02 (Multivariable Calculus). "
            "What is the shortest path to being eligible for 6.3900 (Intro to ML)?"
        ),
        "expected_decision": "PATH",
        "expected_keywords": ["6.1010", "6.1200", "6.1210", "18.06", "6.3900"],
        "abstain_expected": False,
    },

    # ── Category PR: Program Requirements ─────────────────────────────────
    {
        "id": "PR-01",
        "category": "Program Requirement",
        "query": (
            "How many AUS (Advanced Undergraduate Subjects) do I need for "
            "the Course 6-3 degree, and what are the group distribution rules?"
        ),
        "expected_decision": "INFO",
        "expected_keywords": ["3 AUS", "two different groups", "group", "maximum 2"],
        "abstain_expected": False,
    },
    {
        "id": "PR-02",
        "category": "Program Requirement",
        "query": (
            "Is this AUS combination valid: 6.3940 (Group C), 6.4110 (Group C), "
            "and 6.5840 (Group B)?"
        ),
        "expected_decision": "VALID",
        "expected_keywords": ["valid", "Group B", "Group C", "two different groups"],
        "abstain_expected": False,
    },
    {
        "id": "PR-03",
        "category": "Program Requirement",
        "query": (
            "What is the minimum grade I need in Foundation subjects "
            "like 6.1200 to count them toward the Course 6-3 degree?"
        ),
        "expected_decision": "INFO",
        "expected_keywords": ["C or better", "D or F", "retake", "Foundation"],
        "abstain_expected": False,
    },
    {
        "id": "PR-04",
        "category": "Program Requirement",
        "query": (
            "How many Header subjects are required in Course 6-3 "
            "and can you list some examples?"
        ),
        "expected_decision": "INFO",
        "expected_keywords": ["2 Header", "6.3900", "6.4100", "header"],
        "abstain_expected": False,
    },
    {
        "id": "PR-05",
        "category": "Program Requirement",
        "query": (
            "What is the maximum number of units I can take per semester "
            "without special approval, and what happens if I want to exceed that?"
        ),
        "expected_decision": "INFO",
        "expected_keywords": ["57 units", "Dean", "advisor", "approval"],
        "abstain_expected": False,
    },

    # ── Category ND: Not-in-docs / Abstention ──────────────────────────────
    {
        "id": "ND-01",
        "category": "Not in Catalog",
        "query": (
            "Which professor is teaching 6.3900 in Fall 2025 "
            "and what are their office hours?"
        ),
        "expected_decision": "ABSTAIN",
        "expected_keywords": ["don't have", "not in", "registrar", "schedule"],
        "abstain_expected": True,
    },
    {
        "id": "ND-02",
        "category": "Not in Catalog",
        "query": (
            "Will 6.5840 (Distributed Systems) definitely be offered next Spring? "
            "Can you guarantee it?"
        ),
        "expected_decision": "ABSTAIN",
        "expected_keywords": ["cannot guarantee", "not in", "verify", "schedule"],
        "abstain_expected": True,
    },
    {
        "id": "ND-03",
        "category": "Not in Catalog",
        "query": (
            "What is Professor Abelson's specific policy on late homework "
            "for 6.1020 this semester?"
        ),
        "expected_decision": "ABSTAIN",
        "expected_keywords": ["don't have", "not in", "syllabus", "instructor"],
        "abstain_expected": True,
    },
    {
        "id": "ND-04",
        "category": "Not in Catalog",
        "query": (
            "How many students are currently on the waitlist for 6.3900?"
        ),
        "expected_decision": "ABSTAIN",
        "expected_keywords": ["don't have", "not in", "registrar", "real-time"],
        "abstain_expected": True,
    },
    {
        "id": "ND-05",
        "category": "Not in Catalog",
        "query": (
            "Can I get an override for 6.1210 without taking 6.1200 if I "
            "promise to study the material on my own?"
        ),
        "expected_decision": "PARTIAL",   # policy mentions instructor consent but not 'promise'
        "expected_keywords": ["instructor", "consent", "override", "not guaranteed"],
        "abstain_expected": True,  # specific approval process not in docs
    },
]

# ── Scoring helpers ────────────────────────────────────────────────────────

def has_citation(response: str) -> bool:
    """Check if response contains a citation marker."""
    return "[Source:" in response or "Citations:" in response

def check_keywords(response: str, keywords: list) -> float:
    """Return fraction of expected keywords found in response (case-insensitive)."""
    resp_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return found / len(keywords) if keywords else 0.0

def check_abstention(response: str, should_abstain: bool) -> bool:
    """Check if abstention behavior is correct."""
    abstain_phrases = [
        "don't have that information",
        "not in the provided catalog",
        "i don't have",
        "cannot find",
        "not mentioned in",
        "not available in",
        "no information",
        "cannot guarantee",
    ]
    abstained = any(p in response.lower() for p in abstain_phrases)
    return abstained == should_abstain

def score_response(item: dict, response: str) -> dict:
    """Score a single response across multiple dimensions."""
    citation_ok  = has_citation(response)
    keyword_score = check_keywords(response, item["expected_keywords"])
    abstain_ok   = check_abstention(response, item["abstain_expected"])
    has_structure = all(
        marker in response
        for marker in ["Answer", "Why", "Citations", "Assumptions"]
    )
    return {
        "id"            : item["id"],
        "category"      : item["category"],
        "has_citation"  : citation_ok,
        "keyword_score" : round(keyword_score, 2),
        "abstain_ok"    : abstain_ok,
        "structured"    : has_structure,
        "overall_pass"  : citation_ok and abstain_ok and keyword_score >= 0.5,
    }

# ── Main evaluation loop ───────────────────────────────────────────────────

def run_evaluation():
    from src.rag_chain import MITCatalogAssistant

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set in environment.")
        sys.exit(1)

    print("=" * 70)
    print("MIT CS Catalog RAG — Evaluation (25 queries)")
    print("=" * 70)

    assistant = MITCatalogAssistant(groq_api_key=api_key)

    results   = []
    responses = []

    for i, item in enumerate(TEST_SET, 1):
        print(f"\n[{i:02d}/25] {item['id']} — {item['category']}")
        print(f"  Q: {item['query'][:80]}…")

        try:
            response = assistant.ask(item["query"])
            time.sleep(1.2)   # respect Groq rate limits
        except Exception as e:
            print(f"  ERROR: {e}")
            response = f"ERROR: {e}"

        scores = score_response(item, response)
        results.append(scores)
        responses.append({"item": item, "response": response, "scores": scores})

        status = "✅" if scores["overall_pass"] else "❌"
        print(f"  {status} citation={scores['has_citation']} | "
              f"keywords={scores['keyword_score']:.0%} | "
              f"abstain_ok={scores['abstain_ok']} | "
              f"structured={scores['structured']}")

    # ── Aggregate metrics ────────────────────────────────────────────────
    total = len(results)
    citation_rate    = sum(r["has_citation"]  for r in results) / total
    abstain_rate     = sum(r["abstain_ok"]    for r in results) / total
    structure_rate   = sum(r["structured"]    for r in results) / total
    overall_pass     = sum(r["overall_pass"]  for r in results) / total
    avg_keyword      = sum(r["keyword_score"] for r in results) / total

    # Per-category
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, []).append(r["overall_pass"])
    cat_summary = {c: f"{sum(v)/len(v):.0%} ({sum(v)}/{len(v)})" for c, v in cats.items()}

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Citation coverage rate  : {citation_rate:.0%}  ({int(citation_rate*total)}/{total})")
    print(f"  Abstention accuracy     : {abstain_rate:.0%}  ({int(abstain_rate*total)}/{total})")
    print(f"  Structured format rate  : {structure_rate:.0%}  ({int(structure_rate*total)}/{total})")
    print(f"  Avg keyword match rate  : {avg_keyword:.0%}")
    print(f"  Overall pass rate       : {overall_pass:.0%}  ({int(overall_pass*total)}/{total})")
    print()
    for cat, summary in cat_summary.items():
        print(f"  {cat:30s}: {summary}")

    # ── Write full report ────────────────────────────────────────────────
    report_path = Path("outputs/evaluation_report.json")
    report_path.parent.mkdir(exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "llama3-8b-8192 via Groq",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "summary": {
            "citation_coverage_rate": round(citation_rate, 4),
            "abstention_accuracy":    round(abstain_rate, 4),
            "structure_rate":         round(structure_rate, 4),
            "avg_keyword_match":      round(avg_keyword, 4),
            "overall_pass_rate":      round(overall_pass, 4),
            "per_category":           cat_summary,
        },
        "results": [
            {
                "id":       r["item"]["id"],
                "category": r["item"]["category"],
                "query":    r["item"]["query"],
                "response": r["response"],
                "scores":   r["scores"],
            }
            for r in responses
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Full report saved → {report_path}")

    # ── Print 3 example transcripts ─────────────────────────────────────
    print_example_transcripts(responses)

    return report


def print_example_transcripts(responses: list):
    print("\n" + "=" * 70)
    print("EXAMPLE TRANSCRIPTS")
    print("=" * 70)

    # 1. Correct eligibility decision with citations
    pc_pass = next(
        (r for r in responses
         if r["item"]["category"] == "Prerequisite Check" and r["scores"]["overall_pass"]),
        None
    )
    if pc_pass:
        print("\n── TRANSCRIPT 1: Correct Eligibility Decision with Citations ──")
        print(f"Query: {pc_pass['item']['query']}")
        print(f"Response:\n{pc_pass['response'][:1200]}")

    # 2. Course plan output
    plan = next(
        (r for r in responses if r["item"]["id"] == "PCC-01"),
        None
    )
    if plan:
        print("\n── TRANSCRIPT 2: Prerequisite Chain (Multi-Hop) ──")
        print(f"Query: {plan['item']['query']}")
        print(f"Response:\n{plan['response'][:1200]}")

    # 3. Correct abstention
    nd_pass = next(
        (r for r in responses
         if r["item"]["category"] == "Not in Catalog" and r["scores"]["abstain_ok"]),
        None
    )
    if nd_pass:
        print("\n── TRANSCRIPT 3: Correct Abstention + Guidance ──")
        print(f"Query: {nd_pass['item']['query']}")
        print(f"Response:\n{nd_pass['response'][:1200]}")


if __name__ == "__main__":
    run_evaluation()
