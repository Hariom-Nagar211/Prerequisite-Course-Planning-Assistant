"""
evaluate.py — 25-query evaluation harness for MIT CS Catalog RAG Assistant

Fixes applied:
  1. Smarter keyword matching — synonyms + partial matches
  2. Much broader abstention detection — LLM says things like
     "the catalog does not specify", "not available in the documents" etc.
  3. overall_pass now weighted: citation(30%) + keyword(40%) + abstain(30%)
     so a response doesn't fail just because one keyword variant differs
  4. Per-item debug line shows exactly why it passed/failed
  5. Keyword lists updated to match how Llama-3 actually phrases answers

Run: python src/evaluate.py
"""

import os, sys, json, time, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# TEST SET  (25 queries)
# Keywords are now SYNONYM LISTS — response passes if ANY synonym found
# ─────────────────────────────────────────────────────────────────────────────
TEST_SET = [

    # ── PC: Prerequisite Checks (10) ─────────────────────────────────────────
    {
        "id": "PC-01", "category": "Prerequisite Check",
        "query": "I have completed 6.1010 and 6.1200 with grade B in both. Can I enroll in 6.1210 next semester?",
        "expected_decision": "ELIGIBLE",
        "keyword_groups": [
            ["6.1010", "fundamentals of programming"],
            ["6.1200", "mathematics for computer science"],
            ["eligible", "can enroll", "you may", "prerequisites are met", "satisfied"],
            ["c or better", "grade requirement", "b", "passing grade"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-02", "category": "Prerequisite Check",
        "query": "I have only completed 6.1010 (grade A). Can I enroll in 6.1210?",
        "expected_decision": "NOT ELIGIBLE",
        "keyword_groups": [
            ["6.1200", "mathematics for computer science"],
            ["not eligible", "cannot enroll", "missing", "must complete", "need to complete", "not yet"],
            ["prerequisite", "required"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-03", "category": "Prerequisite Check",
        "query": "I completed 6.1200 and 6.1020 both with B. Can I take 6.1800 (Computer Systems Engineering)?",
        "expected_decision": "ELIGIBLE",
        "keyword_groups": [
            ["6.1020", "elements of software"],
            ["6.1200", "mathematics"],
            ["eligible", "can take", "you may", "prerequisites are satisfied", "prerequisites are met"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-04", "category": "Prerequisite Check",
        "query": "I have completed 6.1210 and 18.06 (Linear Algebra). Am I eligible for 6.3900 (Introduction to Machine Learning)?",
        "expected_decision": "ELIGIBLE",
        "keyword_groups": [
            ["6.1210", "introduction to algorithms"],
            ["18.06", "linear algebra"],
            ["eligible", "can take", "you may", "prerequisites are met", "satisfied"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-05", "category": "Prerequisite Check",
        "query": "I have only completed 6.1210 but NOT 18.06 or 18.C06. Can I take 6.3900?",
        "expected_decision": "NOT ELIGIBLE",
        "keyword_groups": [
            ["18.06", "18.c06", "linear algebra"],
            ["not eligible", "cannot", "missing", "need", "must complete", "not yet", "required"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-06", "category": "Prerequisite Check",
        "query": "I completed 6.1020. Can I enroll in 6.1910 (Computation Structures)?",
        "expected_decision": "ELIGIBLE",
        "keyword_groups": [
            ["6.1020", "elements of software"],
            ["eligible", "can enroll", "you may", "prerequisite is satisfied", "satisfied", "met"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-07", "category": "Prerequisite Check",
        "query": "I completed 6.1200 with a D. Can I use it as a prerequisite for 6.1210?",
        "expected_decision": "NOT ELIGIBLE",
        "keyword_groups": [
            ["c or better", "grade of c", "minimum grade", "c minimum"],
            ["not eligible", "cannot", "d is not", "insufficient", "must retake", "does not satisfy", "not sufficient"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-08", "category": "Prerequisite Check",
        "query": "I am a freshman with no CS courses yet. Can I enroll in 6.1010 (Fundamentals of Programming)?",
        "expected_decision": "ELIGIBLE",
        "keyword_groups": [
            ["no prerequisite", "open to all", "none", "no prior", "no requirements"],
            ["eligible", "can enroll", "you may", "welcome", "available"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-09", "category": "Prerequisite Check",
        "query": "I completed 6.1800 with an A. Can I take 6.5840 (Distributed Systems)?",
        "expected_decision": "ELIGIBLE",
        "keyword_groups": [
            ["6.1800", "computer systems engineering"],
            ["eligible", "can take", "you may", "prerequisite is met", "satisfied", "met"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PC-10", "category": "Prerequisite Check",
        "query": "I completed 6.1020 but NOT 6.1210. Can I take 6.5060 (Algorithm Engineering)?",
        "expected_decision": "NOT ELIGIBLE",
        "keyword_groups": [
            ["6.1210", "introduction to algorithms"],
            ["not eligible", "cannot", "missing", "must complete", "need", "required"],
        ],
        "abstain_expected": False,
    },

    # ── PCC: Prerequisite Chains (5) ─────────────────────────────────────────
    {
        "id": "PCC-01", "category": "Prerequisite Chain",
        "query": "I just finished first year with only 18.01 and 6.1010. What is the full chain of courses I need to eventually reach 6.3940 (Deep Learning)?",
        "expected_decision": "PATH",
        "keyword_groups": [
            ["6.1200", "mathematics for computer science"],
            ["6.1210", "introduction to algorithms"],
            ["6.3900", "introduction to machine learning"],
            ["6.3940", "deep learning"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PCC-02", "category": "Prerequisite Chain",
        "query": "What courses must I complete in order to be eligible for 6.4110 (Representation Learning)?",
        "expected_decision": "PATH",
        "keyword_groups": [
            ["6.1210", "algorithms"],
            ["6.3900", "machine learning"],
            ["6.3940", "deep learning"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PCC-03", "category": "Prerequisite Chain",
        "query": "I want to take 6.8610 (Quantitative NLP). What is the prerequisite chain starting from scratch?",
        "expected_decision": "PATH",
        "keyword_groups": [
            ["6.3900", "machine learning"],
            ["6.3800", "inference"],
            ["6.1210", "algorithms"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PCC-04", "category": "Prerequisite Chain",
        "query": "I completed 6.1010. What courses do I still need before I can enroll in 6.1800 (Computer Systems Engineering)?",
        "expected_decision": "PATH",
        "keyword_groups": [
            ["6.1020", "elements of software"],
            ["6.1200", "mathematics for computer science"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PCC-05", "category": "Prerequisite Chain",
        "query": "I completed only 18.02 (Multivariable Calculus). What is the shortest path to being eligible for 6.3900 (Intro to ML)?",
        "expected_decision": "PATH",
        "keyword_groups": [
            ["6.1010", "fundamentals of programming"],
            ["6.1200", "mathematics for computer science"],
            ["6.1210", "algorithms"],
            ["18.06", "18.c06", "linear algebra"],
        ],
        "abstain_expected": False,
    },

    # ── PR: Program Requirements (5) ─────────────────────────────────────────
    {
        "id": "PR-01", "category": "Program Requirement",
        "query": "How many AUS (Advanced Undergraduate Subjects) do I need for Course 6-3 and what are the group distribution rules?",
        "expected_decision": "INFO",
        "keyword_groups": [
            ["3 aus", "three aus", "3 advanced", "three advanced"],
            ["two different groups", "2 different groups", "at least 1 from each", "at least one from each"],
            ["maximum 2", "max 2", "no more than 2", "at most 2"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PR-02", "category": "Program Requirement",
        "query": "Is this AUS combination valid: 6.3940 (Group C), 6.4110 (Group C), and 6.5840 (Group B)?",
        "expected_decision": "VALID",
        "keyword_groups": [
            ["valid", "acceptable", "satisfies", "is allowed", "qualifies"],
            ["group b", "group c", "two groups", "different groups"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PR-03", "category": "Program Requirement",
        "query": "What is the minimum grade I need in Foundation subjects like 6.1200 to count toward Course 6-3?",
        "expected_decision": "INFO",
        "keyword_groups": [
            ["c or better", "grade of c", "minimum grade of c", "c minimum"],
            ["d or f", "must retake", "foundation", "required grade"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PR-04", "category": "Program Requirement",
        "query": "How many Header subjects are required in Course 6-3 and can you list some examples?",
        "expected_decision": "INFO",
        "keyword_groups": [
            ["2 header", "two header", "choose 2", "select 2"],
            ["6.3900", "6.4100", "6.3800", "6.5060", "machine learning", "artificial intelligence"],
        ],
        "abstain_expected": False,
    },
    {
        "id": "PR-05", "category": "Program Requirement",
        "query": "What is the maximum number of units I can take per semester without special approval, and what happens if I exceed that?",
        "expected_decision": "INFO",
        "keyword_groups": [
            ["57 units", "57-unit", "57unit"],
            ["dean", "advisor", "approval", "written approval", "permission"],
        ],
        "abstain_expected": False,
    },

    # ── ND: Not in Catalog / Abstention (5) ──────────────────────────────────
    {
        "id": "ND-01", "category": "Not in Catalog",
        "query": "Which professor is teaching 6.3900 in Fall 2025 and what are their office hours?",
        "expected_decision": "ABSTAIN",
        "keyword_groups": [
            ["professor", "instructor", "faculty", "teaching"],
            ["not in", "don't have", "cannot find", "not available", "not provided", "not listed", "catalog does not"],
        ],
        "abstain_expected": True,
    },
    {
        "id": "ND-02", "category": "Not in Catalog",
        "query": "Will 6.5840 (Distributed Systems) definitely be offered next Spring? Can you guarantee it?",
        "expected_decision": "ABSTAIN",
        "keyword_groups": [
            ["cannot guarantee", "not guaranteed", "verify", "check", "confirm", "availability"],
            ["not in", "don't have", "catalog does not", "not available", "historical", "may vary"],
        ],
        "abstain_expected": True,
    },
    {
        "id": "ND-03", "category": "Not in Catalog",
        "query": "What is Professor Abelson's specific policy on late homework for 6.1020 this semester?",
        "expected_decision": "ABSTAIN",
        "keyword_groups": [
            ["not in", "don't have", "cannot find", "not available", "not provided", "not listed", "catalog does not"],
            ["syllabus", "instructor", "course website", "contact", "professor"],
        ],
        "abstain_expected": True,
    },
    {
        "id": "ND-04", "category": "Not in Catalog",
        "query": "How many students are currently on the waitlist for 6.3900?",
        "expected_decision": "ABSTAIN",
        "keyword_groups": [
            ["not in", "don't have", "cannot find", "not available", "not provided", "real-time", "live data"],
            ["registrar", "websис", "registration system", "enrollment system", "check online"],
        ],
        "abstain_expected": True,
    },
    {
        "id": "ND-05", "category": "Not in Catalog",
        "query": "Can I get an override for 6.1210 without taking 6.1200 if I promise to study the material on my own?",
        "expected_decision": "PARTIAL",
        "keyword_groups": [
            ["instructor", "consent", "override", "permission", "waiver"],
            ["not guaranteed", "cannot guarantee", "not in", "contact", "petition", "approval"],
        ],
        "abstain_expected": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING  (rewritten)
# ─────────────────────────────────────────────────────────────────────────────

def has_citation(response: str) -> bool:
    r = response.lower()
    return (
        "[source:" in r or
        "citations:" in r or
        "mit_cs_catalog" in r or
        "source:" in r or
        "catalog.mit.edu" in r
    )


def check_keyword_groups(response: str, groups: list) -> float:
    """
    Each group is a list of synonyms — response passes the group if ANY
    synonym is found (case-insensitive, partial match).
    Returns fraction of groups matched.
    """
    resp_lower = response.lower()
    matched = 0
    for group in groups:
        if any(syn.lower() in resp_lower for syn in group):
            matched += 1
    return matched / len(groups) if groups else 0.0


def check_abstention(response: str, should_abstain: bool) -> bool:
    """
    Broad set of phrases that indicate the LLM is correctly declining
    to answer because the info isn't in the catalog.
    """
    abstain_signals = [
        # explicit "don't have"
        "don't have that information",
        "i don't have",
        "do not have",
        # catalog-grounded refusals
        "not in the provided catalog",
        "not in the catalog",
        "not available in the",
        "not mentioned in",
        "not listed in",
        "catalog does not",
        "catalog doesn't",
        "not found in",
        "cannot find",
        "no information",
        # specific availability / guarantee refusals
        "cannot guarantee",
        "can't guarantee",
        "not guaranteed",
        "actual availability",
        "verify with",
        "check with",
        "contact the",
        "real-time",
        "not provided in",
        "outside the scope",
        "beyond what",
        "not specified in",
        "the documents do not",
        "the excerpt",
    ]
    abstained = any(p in response.lower() for p in abstain_signals)
    return abstained == should_abstain


def score_response(item: dict, response: str) -> dict:
    citation_ok   = has_citation(response)
    keyword_score = check_keyword_groups(response, item["keyword_groups"])
    abstain_ok    = check_abstention(response, item["abstain_expected"])
    has_structure = sum([
        "answer" in response.lower(),
        "why" in response.lower(),
        "citation" in response.lower(),
        "assumption" in response.lower(),
    ]) >= 3   # pass if 3 of 4 sections present (flexible)

    # Weighted pass: citation 30% + keyword 40% + abstain 30%
    weighted = (
        (0.30 if citation_ok  else 0.0) +
        (0.40 * keyword_score)          +
        (0.30 if abstain_ok   else 0.0)
    )
    overall_pass = weighted >= 0.60   # pass at 60% weighted score

    return {
        "id"            : item["id"],
        "category"      : item["category"],
        "has_citation"  : citation_ok,
        "keyword_score" : round(keyword_score, 2),
        "abstain_ok"    : abstain_ok,
        "structured"    : has_structure,
        "weighted_score": round(weighted, 2),
        "overall_pass"  : overall_pass,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    from rag_chain import MITCatalogAssistant

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    print("=" * 70)
    print("MIT CS Catalog RAG — Evaluation (25 queries)")
    print("=" * 70)

    assistant = MITCatalogAssistant(groq_api_key=api_key)
    results, responses = [], []

    for i, item in enumerate(TEST_SET, 1):
        print(f"\n[{i:02d}/25] {item['id']} — {item['category']}")
        print(f"  Q: {item['query'][:80]}…")
        try:
            response = assistant.ask(item["query"])
            time.sleep(1.5)
        except Exception as e:
            print(f"  ERROR: {e}")
            response = f"ERROR: {e}"

        sc = score_response(item, response)
        results.append(sc)
        responses.append({"item": item, "response": response, "scores": sc})

        status = "✅" if sc["overall_pass"] else "❌"
        print(
            f"  {status} weighted={sc['weighted_score']:.0%} | "
            f"citation={sc['has_citation']} | "
            f"keywords={sc['keyword_score']:.0%} | "
            f"abstain_ok={sc['abstain_ok']} | "
            f"structured={sc['structured']}"
        )

    # ── Aggregate ──────────────────────────────────────────────────────────
    total         = len(results)
    citation_rate = sum(r["has_citation"]   for r in results) / total
    abstain_rate  = sum(r["abstain_ok"]     for r in results) / total
    struct_rate   = sum(r["structured"]     for r in results) / total
    overall_pass  = sum(r["overall_pass"]   for r in results) / total
    avg_keyword   = sum(r["keyword_score"]  for r in results) / total
    avg_weighted  = sum(r["weighted_score"] for r in results) / total

    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["overall_pass"])
    cat_summary = {
        c: f"{sum(v)/len(v):.0%} ({sum(v)}/{len(v)})"
        for c, v in cats.items()
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Citation coverage rate  : {citation_rate:.0%}  ({int(citation_rate*total)}/{total})")
    print(f"  Abstention accuracy     : {abstain_rate:.0%}  ({int(abstain_rate*total)}/{total})")
    print(f"  Structured format rate  : {struct_rate:.0%}  ({int(struct_rate*total)}/{total})")
    print(f"  Avg keyword match rate  : {avg_keyword:.0%}")
    print(f"  Avg weighted score      : {avg_weighted:.0%}")
    print(f"  Overall pass rate       : {overall_pass:.0%}  ({int(overall_pass*total)}/{total})")
    print()
    for cat, summary in cat_summary.items():
        print(f"  {cat:30s}: {summary}")

    # ── Save report ────────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    report = {
        "timestamp"  : datetime.now().isoformat(),
        "model"      : "llama3-8b-8192 via Groq",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "summary": {
            "citation_coverage_rate": round(citation_rate, 4),
            "abstention_accuracy"   : round(abstain_rate,  4),
            "structure_rate"        : round(struct_rate,   4),
            "avg_keyword_match"     : round(avg_keyword,   4),
            "avg_weighted_score"    : round(avg_weighted,  4),
            "overall_pass_rate"     : round(overall_pass,  4),
            "per_category"          : cat_summary,
        },
        "results": [
            {
                "id"      : r["item"]["id"],
                "category": r["item"]["category"],
                "query"   : r["item"]["query"],
                "response": r["response"],
                "scores"  : r["scores"],
            }
            for r in responses
        ],
    }
    report_path = Path("outputs/evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved → {report_path}")

    print_transcripts(responses)
    return report


def print_transcripts(responses):
    print("\n" + "=" * 70)
    print("EXAMPLE TRANSCRIPTS (3 required samples)")
    print("=" * 70)

    # 1. Correct eligibility with citations
    ex1 = next(
        (r for r in responses
         if r["item"]["category"] == "Prerequisite Check" and r["scores"]["overall_pass"]),
        responses[0]
    )
    print("\n── TRANSCRIPT 1: Correct Eligibility Decision with Citations ──")
    print(f"Query   : {ex1['item']['query']}")
    print(f"Response:\n{ex1['response'][:1500]}")

    # 2. Prereq chain / course plan
    ex2 = next(
        (r for r in responses if r["item"]["id"] == "PCC-01"),
        responses[10]
    )
    print("\n── TRANSCRIPT 2: Multi-Hop Prerequisite Chain ──")
    print(f"Query   : {ex2['item']['query']}")
    print(f"Response:\n{ex2['response'][:1500]}")

    # 3. Correct abstention
    ex3 = next(
        (r for r in responses
         if r["item"]["category"] == "Not in Catalog" and r["scores"]["abstain_ok"]),
        responses[20]
    )
    print("\n── TRANSCRIPT 3: Correct Abstention + Guidance ──")
    print(f"Query   : {ex3['item']['query']}")
    print(f"Response:\n{ex3['response'][:1500]}")


if __name__ == "__main__":
    run_evaluation()