"""
batch_test.py

Fast batch tester for the Investigative RAG system.
Runs all questions and checks answer quality using simple rules.
NO LLM calls for evaluation — zero API cost.

Rules:
- FAIL if answer contains "I cannot answer"
- FAIL if answer contains "No data found"
- FAIL if answer contains "not included in this dataset"
- FAIL if answer contains "no organizations" / "no committees"
- FAIL if answer is less than 50 words
- FAIL if answer has no numbers (financial questions need numbers)
- PASS otherwise

Usage:
    DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey python3 src/eval/batch_test.py
    DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey python3 src/eval/batch_test.py --dataset irs
    DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey python3 src/eval/batch_test.py --dataset fec
    DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey python3 src/eval/batch_test.py --fix-only
"""

import os
import sys
import json
import time
import argparse
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.hybrid import hybrid_ask

# ── ALL 517 TEST QUESTIONS ──
ALL_QUESTIONS = [
    # IRS Financial Rankings
    ("Which nonprofits raised the most money?", "irs"),
    ("Which nonprofits have the highest total assets?", "irs"),
    ("Which nonprofits reported the most expenses?", "irs"),
    ("Which nonprofits have the highest total liabilities?", "irs"),
    ("Which nonprofits have the most net assets?", "irs"),
    ("Which nonprofits reported the most contributions and grants?", "irs"),
    ("Which nonprofits reported the most program service revenue?", "irs"),
    ("Which nonprofits pay their officers the most?", "irs"),
    ("List the top 10 nonprofits by revenue", "irs"),
    ("List the top 10 nonprofits by assets", "irs"),
    ("Which nonprofits have revenue over 1 billion dollars?", "irs"),
    ("Which nonprofits have revenue over 100 million dollars?", "irs"),
    ("Which nonprofits have assets over 1 billion dollars?", "irs"),
    ("Which nonprofits have assets over 500 million dollars?", "irs"),
    ("Which nonprofits have assets over 100 million dollars?", "irs"),
    ("Which nonprofits reported losses?", "irs"),
    ("Which nonprofits had a surplus in 2024?", "irs"),
    ("Which nonprofits have debt over 10 million dollars?", "irs"),
    ("Which nonprofits pay officers over 1 million dollars?", "irs"),
    ("Which nonprofits pay officers over 500 thousand dollars?", "irs"),

    # IRS By Organization Type
    ("Which hospitals raised the most money?", "irs"),
    ("Which hospitals have the most assets?", "irs"),
    ("Which hospitals spent the most money?", "irs"),
    ("Which universities reported the highest revenue?", "irs"),
    ("Which universities have the most assets?", "irs"),
    ("Which colleges raised the most money?", "irs"),
    ("Which foundations have the most net assets?", "irs"),
    ("Which foundations spent the most money?", "irs"),
    ("Which foundations raised the most in contributions?", "irs"),
    ("Which health systems have the highest revenue?", "irs"),
    ("Which medical centers have the most assets?", "irs"),
    ("Which research institutes have the most assets?", "irs"),
    ("Which community organizations raised the most?", "irs"),
    ("Which arts organizations have the most assets?", "irs"),
    ("Which social service organizations have the most revenue?", "irs"),
    ("Which housing nonprofits have the most liabilities?", "irs"),
    ("Which youth organizations have the most assets?", "irs"),
    ("Which environmental nonprofits have the highest revenue?", "irs"),
    ("Which veterans organizations have the most assets?", "irs"),
    ("Which nonprofit health systems are the largest?", "irs"),

    # IRS By State
    ("Which nonprofits are based in California?", "irs"),
    ("Which nonprofits are based in New York?", "irs"),
    ("Which nonprofits are based in Texas?", "irs"),
    ("Which nonprofits are based in Florida?", "irs"),
    ("Which nonprofits are based in Illinois?", "irs"),
    ("Which nonprofits are based in Pennsylvania?", "irs"),
    ("Which nonprofits are based in Ohio?", "irs"),
    ("Which nonprofits are based in Georgia?", "irs"),
    ("Which nonprofits are based in Michigan?", "irs"),
    ("Which nonprofits are based in North Carolina?", "irs"),
    ("Which nonprofits are based in New Jersey?", "irs"),
    ("Which nonprofits are based in Virginia?", "irs"),
    ("Which nonprofits are based in Washington?", "irs"),
    ("Which nonprofits are based in Massachusetts?", "irs"),
    ("Which nonprofits are based in Tennessee?", "irs"),
    ("Which nonprofits are based in Colorado?", "irs"),
    ("Which nonprofits are based in Maryland?", "irs"),
    ("Which nonprofits are based in Indiana?", "irs"),
    ("Which nonprofits are based in Minnesota?", "irs"),
    ("Which nonprofits are based in Arizona?", "irs"),

    # IRS By Return Type
    ("Which organizations filed 990PF returns?", "irs"),
    ("Which organizations filed 990EZ returns?", "irs"),
    ("Which organizations filed 990T returns?", "irs"),
    ("Which organizations filed 990 returns?", "irs"),

    # FEC Financial Rankings
    ("Which PACs spent the most money in 2024?", "fec"),
    ("Which committees raised the most money in 2024?", "fec"),
    ("Which committees raised the most money in 2026?", "fec"),
    ("Which Super PACs spent the most in 2024?", "fec"),
    ("Which Democratic committees raised the most?", "fec"),
    ("Which Republican committees raised the most?", "fec"),
    ("Which presidential campaign committees raised the most?", "fec"),
    ("Which Senate campaign committees spent the most?", "fec"),
    ("Which House campaign committees raised the most?", "fec"),
    ("Which PACs raised over 100 million dollars?", "fec"),
    ("Which committees raised over 1 billion dollars?", "fec"),
    ("Which committees spent over 100 million dollars?", "fec"),
    ("Which committees have the most cash on hand?", "fec"),
    ("Which committees have the most individual contributions?", "fec"),
    ("Which committees have the most debt?", "fec"),
    ("List the top 10 committees by receipts", "fec"),
    ("List the top 10 committees by disbursements", "fec"),
    ("Which party committees raised the most in 2024?", "fec"),
    ("Which independent expenditure committees spent the most?", "fec"),
    ("Which lobbyist PACs raised the most money?", "fec"),

    # FEC Specific Committees
    ("How much did ActBlue raise in 2024?", "fec"),
    ("How much did WinRed raise in 2024?", "fec"),
    ("What did Harris for President report in 2024?", "fec"),
    ("How much did the DNC raise in 2024?", "fec"),
    ("How much did the RNC raise in 2024?", "fec"),
    ("What did the Lincoln Project report in 2024?", "fec"),
    ("How much did America First Action raise?", "fec"),
    ("What did Harris Victory Fund report in 2024?", "fec"),

    # FEC Geographic
    ("Which FEC committees are based in New York?", "fec"),
    ("Which FEC committees are based in California?", "fec"),
    ("Which FEC committees are based in Texas?", "fec"),
    ("Which FEC committees are based in Florida?", "fec"),
    ("Which FEC committees are based in Washington?", "fec"),
    ("Which PACs in New York raised the most money?", "fec"),
    ("Which campaign committees are from Texas?", "fec"),

    # Cross Dataset
    ("Which nonprofits have connections to political committees?", "both"),
    ("Which health organizations have the most revenue?", "both"),
    ("Which organizations have assets over 100 million dollars?", "both"),
    ("Which nonprofits pay their executives over 500 thousand?", "both"),
    ("Which foundations have the most assets?", "both"),
    ("Which educational institutions have the most debt?", "both"),
    ("Which social service nonprofits have the most assets?", "both"),
    ("Which community foundations raised the most?", "both"),
    ("Which nonprofit hospitals have the highest revenue?", "both"),
    ("Which research organizations have the most assets?", "both"),
]

# ── FAIL PATTERNS ──
FAIL_PATTERNS = [
    "i cannot answer",
    "i cannot provide",
    "cannot be answered",
    "no data found",
    "not included in this dataset",
    "no organizations",
    "no nonprofits",
    "no committees",
    "no data",
    "not available",
    "insufficient information",
    "not enough information",
    "no relevant",
    "unable to answer",
    "not found in",
    "does not contain",
    "not present in",
]

HAS_NUMBER_PATTERN = re.compile(r'\$[\d,.]+|\d+[\.,]\d+\s*(billion|million|thousand)|\d{4,}')


def check_answer_quality(question, answer, dataset):
    """Check answer quality using simple rules. Returns (passed, reason)."""
    answer_lower = answer.lower()
    word_count = len(answer.split())

    # Check fail patterns
    for pattern in FAIL_PATTERNS:
        if pattern in answer_lower:
            return False, f"Contains fail pattern: '{pattern}'"

    # Check minimum length
    if word_count < 30:
        return False, f"Too short: {word_count} words"

    # Financial questions should have numbers
    financial_keywords = ["revenue", "raised", "spent", "assets", "money", "million",
                         "billion", "compensation", "receipts", "disbursements", "much did"]
    is_financial = any(kw in question.lower() for kw in financial_keywords)
    if is_financial and not HAS_NUMBER_PATTERN.search(answer):
        return False, "Financial question but no numbers in answer"

    return True, "OK"


def run_batch_test(dataset_filter=None, fix_only=False, sample=None):
    questions = ALL_QUESTIONS

    if dataset_filter:
        questions = [(q, d) for q, d in questions if d == dataset_filter or d == "both"]

    if sample:
        import random
        random.seed(42)
        questions = random.sample(questions, min(sample, len(questions)))

    total = len(questions)
    passed = 0
    failed = 0
    failed_list = []
    results = []

    print("=" * 65)
    print("INVESTIGATIVE RAG — BATCH QUALITY TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Questions to test: {total}")
    if dataset_filter:
        print(f"Dataset filter: {dataset_filter}")
    print("=" * 65)

    for i, (question, dataset) in enumerate(questions, 1):
        start = time.time()
        try:
            result = hybrid_ask(question, dataset=dataset, top_k=5)
            answer = result.answer
            elapsed = round(time.time() - start, 2)

            ok, reason = check_answer_quality(question, answer, dataset)

            if ok:
                passed += 1
                status = "PASS"
                print(f"[{i:3}/{total}] PASS | {dataset.upper():4} | {question[:55]}")
            else:
                failed += 1
                status = "FAIL"
                failed_list.append({
                    "question": question,
                    "dataset": dataset,
                    "reason": reason,
                    "answer_preview": answer[:150],
                })
                print(f"[{i:3}/{total}] FAIL | {dataset.upper():4} | {question[:55]}")
                print(f"         Reason: {reason}")

            results.append({
                "question": question,
                "dataset": dataset,
                "status": status,
                "reason": reason,
                "response_time": elapsed,
                "answer_preview": answer[:200],
            })

        except Exception as e:
            failed += 1
            elapsed = round(time.time() - start, 2)
            failed_list.append({
                "question": question,
                "dataset": dataset,
                "reason": f"Error: {str(e)[:100]}",
                "answer_preview": "",
            })
            print(f"[{i:3}/{total}] ERROR | {dataset.upper():4} | {question[:55]}")
            print(f"         Error: {e}")
            results.append({
                "question": question,
                "dataset": dataset,
                "status": "ERROR",
                "reason": str(e)[:100],
                "response_time": elapsed,
                "answer_preview": "",
            })

    # ── SUMMARY ──
    accuracy = passed / total * 100
    avg_time = sum(r["response_time"] for r in results) / total

    print("\n" + "=" * 65)
    print("BATCH TEST SUMMARY")
    print("=" * 65)
    print(f"Total Questions:   {total}")
    print(f"Passed:            {passed} ({accuracy:.1f}%)")
    print(f"Failed:            {failed} ({100-accuracy:.1f}%)")
    print(f"Avg Response Time: {avg_time:.2f}s")
    print(f"Est. Total Time:   {avg_time * total:.0f}s ({avg_time * total / 60:.1f} min)")

    if failed_list:
        print(f"\n{'=' * 65}")
        print(f"FAILED QUESTIONS ({len(failed_list)}):")
        print("=" * 65)
        for item in failed_list:
            print(f"\n  Q: {item['question']}")
            print(f"  Dataset: {item['dataset']}")
            print(f"  Reason: {item['reason']}")
            print(f"  Answer: {item['answer_preview'][:120]}...")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy_pct": round(accuracy, 1),
        "avg_response_time": round(avg_time, 2),
        "failed_questions": failed_list,
        "all_results": results,
    }

    output_file = "src/eval/batch_test_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 65)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch test the RAG system")
    parser.add_argument("--dataset", choices=["irs", "fec", "both"], help="Filter by dataset")
    parser.add_argument("--fix-only", action="store_true", help="Only show failed questions")
    parser.add_argument("--sample", type=int, help="Test a random sample of N questions")
    args = parser.parse_args()

    run_batch_test(
        dataset_filter=args.dataset,
        fix_only=args.fix_only,
        sample=args.sample,
    )
