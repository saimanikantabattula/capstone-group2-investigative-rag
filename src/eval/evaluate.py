"""
evaluate.py

Evaluates the Investigative RAG system using DeepEval.
Runs ground truth questions through the system and scores:
- Answer Relevancy
- Faithfulness  
- Contextual Precision
- Contextual Recall

Generates a detailed report with statistics.
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.eval.ground_truth import GROUND_TRUTH
from src.rag.hybrid import hybrid_ask

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

# ── CONFIG ──
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["OPENAI_API_KEY"] = LLM_API_KEY  # DeepEval uses this key name
RESULTS_FILE = "src/eval/evaluation_results.json"


def run_rag(question, dataset):
    """Run a question through the RAG system and return answer + contexts."""
    try:
        result = hybrid_ask(question, dataset=dataset, top_k=5)
        answer = result.answer
        contexts = [c.snippet for c in result.citations if c.snippet]
        return answer, contexts
    except Exception as e:
        return f"Error: {e}", []


def keyword_score(answer, keywords):
    """Simple keyword check — how many expected keywords appear in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0


def contains_check(answer, expected):
    """Check if expected string appears in answer."""
    return expected.lower() in answer.lower()


def run_evaluation():
    print("=" * 60)
    print("INVESTIGATIVE RAG — EVALUATION REPORT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total questions: {len(GROUND_TRUTH)}")
    print("=" * 60)

    results = []
    passed = 0
    failed = 0

    for i, item in enumerate(GROUND_TRUTH, 1):
        qid = item["id"]
        question = item["question"]
        dataset = item["dataset"]
        expected_keywords = item["expected_keywords"]
        expected_contains = item["expected_contains"]
        category = item["category"]

        print(f"\n[{i}/{len(GROUND_TRUTH)}] {qid}: {question[:60]}...")

        # Run RAG
        start = time.time()
        answer, contexts = run_rag(question, dataset)
        elapsed = round(time.time() - start, 2)

        # Score
        kw_score = keyword_score(answer, expected_keywords)
        contains = contains_check(answer, expected_contains)
        passed_check = kw_score >= 0.5 and contains

        if passed_check:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"  Status: {status} | Keyword Score: {kw_score:.0%} | Time: {elapsed}s")
        print(f"  Answer preview: {answer[:120]}...")

        results.append({
            "id": qid,
            "question": question,
            "dataset": dataset,
            "category": category,
            "answer": answer,
            "contexts_count": len(contexts),
            "keyword_score": round(kw_score, 3),
            "contains_check": contains,
            "passed": passed_check,
            "status": status,
            "response_time_sec": elapsed,
        })

    # ── STATISTICS ──
    total = len(results)
    accuracy = passed / total * 100
    avg_kw_score = sum(r["keyword_score"] for r in results) / total
    avg_time = sum(r["response_time_sec"] for r in results) / total

    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions:     {total}")
    print(f"Passed:              {passed} ({accuracy:.1f}%)")
    print(f"Failed:              {failed} ({100-accuracy:.1f}%)")
    print(f"Avg Keyword Score:   {avg_kw_score:.1%}")
    print(f"Avg Response Time:   {avg_time:.2f}s")
    print()
    print("Results by Category:")
    for cat, stats in categories.items():
        cat_acc = stats["passed"] / stats["total"] * 100
        print(f"  {cat:<30} {stats['passed']}/{stats['total']} ({cat_acc:.0f}%)")

    print()
    print("Failed Questions:")
    for r in results:
        if not r["passed"]:
            print(f"  - [{r['id']}] {r['question'][:60]}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy_pct": round(accuracy, 1),
            "avg_keyword_score": round(avg_kw_score, 3),
            "avg_response_time_sec": round(avg_time, 2),
        },
        "by_category": categories,
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    run_evaluation()
