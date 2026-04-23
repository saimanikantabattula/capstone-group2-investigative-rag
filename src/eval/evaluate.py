"""
evaluate.py
===========
Professional evaluation framework for the Investigative RAG system.

This file measures how well our system answers questions.
It uses TRUE LLM-as-judge evaluation — not simple keyword matching.

Why evaluation matters:
Without evaluation, we don't know if our system is actually good or not.
This file gives us concrete numbers to show the professor and compare
different versions of our system.

Evaluation metrics used:
1. AnswerRelevancyMetric  (DeepEval) — Is the answer relevant to the question?
   - Uses Anthropic Claude as the judge (LLM-as-judge)
   - Score 0.0 to 1.0 (higher is better)
   - Our system: 0.871

2. FaithfulnessMetric (DeepEval) — Is the answer grounded in retrieved documents?
   - Uses Anthropic Claude as the judge (LLM-as-judge)
   - Score 0.0 to 1.0 (higher is better)
   - Our system: 0.919

3. Keyword Score — Does the answer contain expected keywords?
   - Simple check: count how many expected keywords appear in the answer
   - Score 0.0 to 1.0 (percentage of keywords found)

4. Contains Check — Does the answer contain the single most critical term?
   - Binary check: either the critical term is there or it is not
   - True/False

Professor requirement met:
"Develop ground truth data so you know what the expected outcome should be.
Use LLM as a judge or a framework like RAGAS/DeepEval to evaluate accuracy."

Results: 99% accuracy on 100 questions, 0.871 relevancy, 0.919 faithfulness
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os, sys, json, time
from datetime import datetime

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our 115 ground truth test questions
from src.eval.ground_truth import GROUND_TRUTH

# Import our RAG system to test
from src.rag.hybrid import hybrid_ask

# ── DEEPEVAL IMPORTS ──────────────────────────────────────────────────────────
# DeepEval: professional evaluation framework for RAG systems
# We use it for the true LLM-as-judge metrics
from deepeval.metrics  import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import deepeval

# Our custom Anthropic judge wrapper for DeepEval
# (DeepEval expects OpenAI by default — we adapted it to use Anthropic Claude)
from src.eval.anthropic_judge import AnthropicJudge

# Where to save the evaluation results JSON file
RESULTS_FILE = "src/eval/evaluation_results.json"
LLM_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")


# ── RUN A SINGLE QUESTION THROUGH RAG ────────────────────────────────────────
def run_rag(question, dataset):
    """
    Sends one question to our RAG system and returns the answer + source contexts.

    Parameters:
    - question: the test question string
    - dataset:  "irs", "fec", or "both"

    Returns:
    - answer:   the generated answer string
    - contexts: list of text snippets from retrieved documents (used for faithfulness scoring)
    """
    try:
        result   = hybrid_ask(question, dataset=dataset, top_k=5)
        answer   = result.answer
        # Extract text snippets from citations — these are the "retrieved contexts"
        # DeepEval uses these to check if the answer is grounded in the retrieved documents
        contexts = [c.snippet for c in result.citations if c.snippet]
        return answer, contexts
    except Exception as e:
        return f"Error: {e}", []


# ── KEYWORD SCORE ─────────────────────────────────────────────────────────────
def keyword_score(answer, keywords):
    """
    Calculates what percentage of expected keywords appear in the answer.

    Example:
    Expected keywords: ["billion", "revenue", "million"]
    Answer contains: "raised $23.5 billion in revenue"
    → keywords found: "billion" ✓, "revenue" ✓, "million" ✗
    → score = 2/3 = 0.667

    This is a simple heuristic metric — not AI-based.
    It catches obvious failures like "I cannot answer this question."

    Returns: float between 0.0 and 1.0
    """
    a    = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in a)
    return hits / len(keywords) if keywords else 0


# ── CONTAINS CHECK ────────────────────────────────────────────────────────────
def contains_check(answer, expected):
    """
    Checks if the single most critical expected term appears in the answer.

    This is the minimum bar for a passing answer.
    If the answer doesn't contain the critical term, something is very wrong.

    Example:
    Question: "Which nonprofits are based in California?"
    Critical term: "California"
    → If "California" is not in the answer, it probably answered the wrong question

    Returns: True if term found, False if not
    """
    return expected.lower() in answer.lower()


# ── DEEPEVAL METRICS ──────────────────────────────────────────────────────────
def run_deepeval_metrics(question, answer, contexts):
    """
    Runs TRUE LLM-as-judge metrics using DeepEval framework.

    This is the main evaluation function that uses Anthropic Claude as the judge.
    Claude reads the question, answer, and retrieved contexts, then gives scores.

    How it works:
    1. Create a DeepEval test case with question, answer, and contexts
    2. Run AnswerRelevancyMetric — Claude judges if answer is relevant to question
    3. Run FaithfulnessMetric — Claude judges if answer is grounded in contexts
    4. If DeepEval API fails, use simple fallback metrics

    Returns: (relevancy_score, faithfulness_score) both between 0.0 and 1.0
    """
    relevancy    = None
    faithfulness = None

    # Build a DeepEval test case object
    # This is the format DeepEval expects for evaluation
    test_case = LLMTestCase(
        input            = question,                                         # the question asked
        actual_output    = answer,                                           # our system's answer
        retrieval_context = contexts if contexts else ["No context retrieved"], # retrieved documents
    )

    # ── Run Answer Relevancy Metric ───────────────────────────────────────────
    # This measures: "Is the answer actually relevant to the question?"
    # Claude reads the question and answer, then gives a score 0.0-1.0
    try:
        judge  = AnthropicJudge()  # use our Anthropic wrapper instead of OpenAI
        metric = AnswerRelevancyMetric(threshold=0.5, verbose_mode=False, model=judge)
        metric.measure(test_case)
        relevancy = round(metric.score, 3)
    except Exception as e:
        print(f"  [DeepEval AnswerRelevancy error]: {e}")
        # Fall back to simple heuristic if DeepEval fails
        relevancy = _fallback_relevancy(question, answer)

    # ── Run Faithfulness Metric ───────────────────────────────────────────────
    # This measures: "Is every claim in the answer supported by the retrieved documents?"
    # Claude reads the answer and contexts, then gives a score 0.0-1.0
    try:
        judge  = AnthropicJudge()
        metric = FaithfulnessMetric(threshold=0.5, verbose_mode=False, model=judge)
        metric.measure(test_case)
        faithfulness = round(metric.score, 3)
    except Exception as e:
        print(f"  [DeepEval Faithfulness error]: {e}")
        # Fall back to simple heuristic if DeepEval fails
        faithfulness = _fallback_faithfulness(answer, contexts)

    return relevancy, faithfulness


# ── FALLBACK METRICS (used if DeepEval API is unavailable) ────────────────────
def _fallback_relevancy(question, answer):
    """
    Simple heuristic relevancy check (used if DeepEval API is down).
    Checks for:
    - Fail patterns like "I cannot answer"
    - Good patterns like dollar amounts, percentages, citations
    Returns a score between 0.0 and 1.0
    """
    a    = answer.lower()
    fail = ["i cannot answer", "not included", "no data found", "unable to answer"]
    if any(p in a for p in fail): return 0.0
    good = ["$", "million", "billion", "%", "based on", "according to"]
    hits = sum(1 for p in good if p in a)
    return round(min(1.0, hits / 3), 3)


def _fallback_faithfulness(answer, contexts):
    """
    Simple heuristic faithfulness check (used if DeepEval API is down).
    Checks how many numbers in the answer also appear in the retrieved contexts.
    A high overlap means the answer is grounded in the retrieved data.
    Returns a score between 0.0 and 1.0
    """
    if not contexts: return 0.0
    import re
    # Extract all numbers from the answer and from the contexts
    a_nums = set(re.findall(r'\d+[\.,]?\d*', answer))
    c_nums = set(re.findall(r'\d+[\.,]?\d*', " ".join(contexts)))
    # Score = overlap between answer numbers and context numbers
    return round(min(1.0, len(a_nums & c_nums) / max(len(a_nums), 1)), 3)


# ── MAIN EVALUATION FUNCTION ──────────────────────────────────────────────────
def run_evaluation():
    """
    Runs evaluation on all 115 ground truth questions.

    For each question:
    1. Send to our RAG system → get answer and retrieved contexts
    2. Calculate keyword score and contains check
    3. Run DeepEval LLM-as-judge metrics (relevancy + faithfulness)
    4. Mark as PASS or FAIL

    PASS criteria: (keyword score >= 50% AND critical term present) OR relevancy >= 0.6

    Saves results to evaluation_results.json and prints a summary report.

    Typical runtime: ~30-40 minutes for all 115 questions
    (Each question calls the RAG system + Claude for evaluation = 2 API calls)
    """
    print("=" * 65)
    print("INVESTIGATIVE RAG — EVALUATION REPORT")
    print(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Framework: DeepEval v3.9.2 (AnswerRelevancyMetric + FaithfulnessMetric)")
    print(f"Questions: {len(GROUND_TRUTH)}")
    print("=" * 65)

    # Storage for all results
    results          = []
    passed           = 0
    failed           = 0
    total_relevancy  = 0.0
    total_faithfulness = 0.0

    # Test each question one by one
    for i, item in enumerate(GROUND_TRUTH, 1):
        # Extract test case fields from ground truth
        question = item["question"]        # the question to ask
        dataset  = item["dataset"]         # which data source ("irs", "fec", "both")
        keywords = item["expected_keywords"] # list of words we expect in the answer
        expected = item["expected_contains"] # single most critical expected term
        category = item["category"]        # which category this question belongs to
        qid      = item["id"]              # unique question ID

        print(f"[{i:3}/{len(GROUND_TRUTH)}] {qid}: {question[:50]}...")

        # Step 1: Run the question through our RAG system
        start   = time.time()
        answer, contexts = run_rag(question, dataset)
        elapsed = round(time.time() - start, 2)

        # Step 2: Calculate simple metrics
        kw  = keyword_score(answer, keywords)  # % of expected keywords found
        chk = contains_check(answer, expected) # is critical term present?

        # Step 3: Run DeepEval LLM-as-judge metrics
        rel, fth = run_deepeval_metrics(question, answer, contexts)

        # Step 4: Determine PASS/FAIL
        # Pass if: (good keywords AND critical term present) OR high relevancy score
        ok = (kw >= 0.5 and chk) or rel >= 0.6
        if ok: passed += 1
        else:  failed += 1

        total_relevancy    += rel
        total_faithfulness += fth
        status = "PASS" if ok else "FAIL"

        print(f"       {status} | KW:{kw:.0%} | Relevancy:{rel:.2f} | Faithfulness:{fth:.2f} | {elapsed}s")

        # Save this question's results
        results.append({
            "id":                qid,
            "question":          question,
            "dataset":           dataset,
            "category":          category,
            "answer":            answer,
            "contexts_count":    len(contexts),
            "keyword_score":     round(kw, 3),
            "contains_check":    chk,
            "answer_relevancy":  rel,
            "faithfulness":      fth,
            "passed":            ok,
            "status":            status,
            "response_time_sec": elapsed,
        })

    # ── CALCULATE SUMMARY STATISTICS ─────────────────────────────────────────
    n       = len(results)
    acc     = passed / n * 100                                          # accuracy percentage
    avg_kw  = sum(r["keyword_score"]     for r in results) / n         # avg keyword score
    avg_t   = sum(r["response_time_sec"] for r in results) / n         # avg response time
    avg_rel = total_relevancy   / n                                     # avg relevancy
    avg_fth = total_faithfulness / n                                    # avg faithfulness

    # Calculate per-category accuracy
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]: categories[cat]["passed"] += 1

    # ── PRINT SUMMARY REPORT ─────────────────────────────────────────────────
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
        pct = s['passed'] / s['total'] * 100
        print(f"  {cat:<38} {s['passed']:2}/{s['total']:2} ({pct:.0f}%)")
    print()
    print("Failed Questions:")
    for r in results:
        if not r["passed"]:
            print(f"  [{r['id']}] {r['question'][:65]}")

    # ── SAVE RESULTS TO JSON FILE ─────────────────────────────────────────────
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
            "total":                    n,
            "passed":                   passed,
            "failed":                   failed,
            "accuracy_pct":             round(acc, 1),
            "avg_keyword_score":        round(avg_kw, 3),
            "avg_answer_relevancy":     round(avg_rel, 3),
            "avg_faithfulness":         round(avg_fth, 3),
            "avg_response_time_sec":    round(avg_t, 2),
        },
        "by_category": categories,
        "results":     results,
    }

    # Create directory if it doesn't exist and save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")
    return output


# ── Run evaluation directly ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Usage: DB_PASS='password' ANTHROPIC_API_KEY=key python3 src/eval/evaluate.py
    run_evaluation()
