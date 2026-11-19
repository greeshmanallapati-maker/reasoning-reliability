# Professor demonstration outline

1. **Research question and falsifiable hypotheses** — explain that CoT is not presumed superior.
2. **Experimental control** — one shared model, fixed native-thinking mode, identical cached questions, deterministic primary run.
3. **Four workflows** — show one-call direct/CoT, genuine two-call verification, and adaptive preliminary-call reuse/routing.
4. **Live question** — run “Compare All” in Streamlit and point out answer, route, latency, tokens, and verification fields; do not claim correctness without a reference.
5. **Data integrity** — open a cached selection and a question-level JSONL record; show model metadata and failure fields.
6. **Evaluation** — distinguish accuracy, consistency, uncertainty proxy, calibration, and cost.
7. **Verification audit** — show wrong→right and right→wrong counts.
8. **Adaptive contribution** — compare difficulty-only, uncertainty-only, and combined routing distributions and utility.
9. **Statistics and plots** — present paired confidence intervals, McNemar test, dataset breakdown, and Pareto frontier.
10. **Honest conclusion** — state only conclusions supported by the completed run and close with threats to validity.

Suggested live commands:

```bash
pytest -q
python scripts/run_benchmark.py --config configs/smoke.yaml --ablations
python scripts/run_analysis.py results/YOUR_RUN_ID
streamlit run app/streamlit_app.py
```
