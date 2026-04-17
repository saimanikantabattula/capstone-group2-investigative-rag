# Evaluation Framework

## Files

### ground_truth.py
100 hand-crafted test questions covering all 7 categories. Each question has expected keywords, a critical expected term, dataset label, and category label.

### evaluate.py
Main evaluation script. Runs all 100 questions and scores using 4 metrics:
- Keyword Score
- Contains Check
- Answer Relevancy (DeepEval LLM-as-judge)
- Faithfulness (DeepEval LLM-as-judge)

Run: DB_PASS='password' ANTHROPIC_API_KEY=key python3 src/eval/evaluate.py

### batch_test.py
Extended test across 109 questions using rule-based quality checks.

Run: DB_PASS='password' ANTHROPIC_API_KEY=key python3 src/eval/batch_test.py

### anthropic_judge.py
Custom DeepEval model wrapper using Anthropic Claude as LLM judge.

## Results
- Accuracy: 99% (99/100 questions)
- Answer Relevancy: 0.871 / 1.0
- Faithfulness: 0.919 / 1.0
- Avg Response Time: 3.22 seconds
