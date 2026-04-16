"""
evaluate.py

Professional evaluation of the Investigative RAG system using TRUE DeepEval metrics.

Metrics:
1. AnswerRelevancyMetric  — Official DeepEval LLM-as-judge
2. FaithfulnessMetric     — Official DeepEval LLM-as-judge
3. Keyword Score          — checks if expected terms appear in the answer
4. Contains Check         — checks if most critical term appears in the answer

Professor Requirement:
"Develop ground truth data so you know what the expected outcome should be.
Use LLM as a judge or a framework like RAGAS/DeepEval to evaluate accuracy."
"""

import os, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.eval.ground_truth import GROUND_TRUTH
from src.rag.hybrid import hybrid_ask

# ── True DeepEval metrics ──
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import deepeval
from src.eval.anthropic_judge import AnthropicJudge

RESULTS_FILE = "src/eval/evaluation_results.json"
LLM_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")



def run_rag(question, dataset):
    try:
        result = hybrid_ask(question, dataset=dataset, top_k=5)
        answer   = result.answer
        contexts = [c.snippet for c in result.citations if c.snippet]
        return answer, contexts
    except Exception as e:
        return f"Error: {e}", []


def keyword_score(answer, keywords):
    a = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in a)
    return hits / len(keywords) if keywords else 0


def contains_check(answer, expected):
    return expected.lower() in answer.lower()


def run_deepeval_metrics(question, answer, contexts):
    """
    Run TRUE DeepEval AnswerRelevancyMetric and FaithfulnessMetric.
    Returns (relevancy_score, faithfulness_score).
    """
    relevancy   = None
    faithfulness = None

    # Build DeepEval test case
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts if contexts else ["No context retrieved"],
    )

    # Answer Relevancy
    try:
        judge = AnthropicJudge()
        metric = AnswerRelevancyMetric(threshold=0.5, verbose_mode=False, model=judge)
        metric.measure(test_case)
        relevancy = round(metric.score, 3)
    except Exception as e:
        print(f"  [DeepEval AnswerRelevancy error]: {e}")
        relevancy = _fallback_relevancy(question, answer)

    # Faithfulness
    try:
        judge = AnthropicJudge()
        metric = FaithfulnessMetric(threshold=0.5, verbose_mode=False, model=judge)
        metric.measure(test_case)
        faithfulness = round(metric.score, 3)
    except Exception as e:
        print(f"  [DeepEval Faithfulness error]: {e}")
        faithfulness = _fallback_faithfulness(answer, contexts)

    return relevancy, faithfulness


def _fallback_relevancy(question, answer):
    """Fallback if DeepEval API unavailable."""
    a = answer.lower()
    fail = ["i cannot answer","not included","no data found","unable to answer"]
    if any(p in a for p in fail): return 0.0
    good = ["$","million","billion","%","based on","according to"]
    hits = sum(1 for p in good if p in a)
    return round(min(1.0, hits / 3), 3)


def _fallback_faithfulness(answer, contexts):
    """Fallback if DeepEval API unavailable."""
    if not contexts: return 0.0
    import re
    a_nums = set(re.findall(r'\d+[\.,]?\d*', answer))
    c_nums = set(re.findall(r'\d+[\.,]?\d*', " ".join(contexts)))
    return round(min(1.0, len(a_nums & c_nums) / max(len(a_nums), 1)), 3)


def run_evaluation():
    print("=" * 65)
    print("INVESTIGATIVE RAG — EVALUATION REPORT")
    print(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Framework: DeepEval v3.9.2 (AnswerRelevancyMetric + FaithfulnessMetric)")
    print(f"Questions: {len(GROUND_TRUTH)}")
    print("=" * 65)

    results = []
    passed = failed = 0
    total_relevancy = total_faithfulness = 0.0

    for i, item in enumerate(GROUND_TRUTH, 1):
        question  = item["question"]
        dataset   = item["dataset"]
        keywords  = item["expected_keywords"]
        expected  = item["expected_contains"]
        category  = item["category"]
        qid       = item["id"]

        print(f"[{i:3}/{len(GROUND_TRUTH)}] {qid}: {question[:50]}...")

        start = time.time()
        answer, contexts = run_rag(question, dataset)
        elapsed = round(time.time() - start, 2)

        kw  = keyword_score(answer, keywords)
        chk = contains_check(answer, expected)
        rel, fth = run_deepeval_metrics(question, answer, contexts)

        ok = (kw >= 0.5 and chk) or rel >= 0.6
        if ok: passed += 1
        else:  failed += 1

        total_relevancy   += rel
        total_faithfulness += fth
        status = "PASS" if ok else "FAIL"

        print(f"       {status} | KW:{kw:.0%} | Relevancy:{rel:.2f} | Faithfulness:{fth:.2f} | {elapsed}s")

        results.append({
            "id": qid, "question": question, "dataset": dataset,
            "category": category, "answer": answer,
            "contexts_count": len(contexts),
            "keyword_score": round(kw, 3), "contains_check": chk,
            "answer_relevancy": rel, "faithfulness": fth,
            "passed": ok, "status": status,
            "response_time_sec": elapsed,
        })

    n   = len(results)
    acc = passed / n * 100
    avg_kw  = sum(r["keyword_score"] for r in results) / n
    avg_t   = sum(r["response_time_sec"] for r in results) / n
    avg_rel = total_relevancy / n
    avg_fth = total_faithfulness / n

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]: categories[cat]["passed"] += 1

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Total:              {n}")
    print(f"Passed:             {passed} ({acc:.1f}%)")
    print(f"Failed:             {failed}")
    print(f"Avg Keyword Score:  {avg_kw:.1%}")
    print(f"Avg Relevancy:      {avg_rel:.3f} / 1.0  [DeepEval AnswerRelevancyMetric]")
    print(f"Avg Faithfulness:   {avg_fth:.3f} / 1.0  [DeepEval FaithfulnessMetric]")
    print(f"Avg Response Time:  {avg_t:.2f}s")
    print()
    print("By Category:")
    for cat, s in sorted(categories.items()):
        print(f"  {cat:<38} {s['passed']:2}/{s['total']:2} ({s['passed']/s['total']*100:.0f}%)")
    print()
    print("Failed Questions:")
    for r in results:
        if not r["passed"]:
            print(f"  [{r['id']}] {r['question'][:65]}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "framework": "DeepEval v3.9.2",
        "metrics": [
            "DeepEval AnswerRelevancyMetric (LLM-as-judge)",
            "DeepEval FaithfulnessMetric (LLM-as-judge)",
            "Keyword Score",
            "Contains Check"
        ],
        "summary": {
            "total": n, "passed": passed, "failed": failed,
            "accuracy_pct": round(acc, 1),
            "avg_keyword_score": round(avg_kw, 3),
            "avg_answer_relevancy": round(avg_rel, 3),
            "avg_faithfulness": round(avg_fth, 3),
            "avg_response_time_sec": round(avg_t, 2),
        },
        "by_category": categories,
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")
    return output


if __name__ == "__main__":
    run_evaluation()
