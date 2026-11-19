# Requirements compliance audit

Audit date: 2026-11-18

## Satisfied

- One shared Hugging Face causal-LM engine; CUDA, Apple MPS, and CPU selection; FP32/FP16/BF16; optional CUDA 4-bit; configurable padded batching for compatible one-call workflows; warm-up; new-token decoding; token log-probabilities; OOM/failure records; software/hardware metadata.
- Prompt-based CoT is separated from Qwen3 native thinking. Primary configs disable native thinking; `native_thinking_smoke.yaml` is separately labeled.
- Official GSM8K, ARC-Easy, and ARC-Challenge loading; robust reference/prediction parsing; deterministic sampling; disjoint development/evaluation pools; cached exact IDs; duplicate safeguards.
- Direct, CoT, genuine two-call verification, and adaptive strategies. Verification never receives ground truth and records corrections, harms, preservation, latency, tokens, and initial/final answers.
- Interpretable difficulty features, likelihood-proxy uncertainty with explicit limitations, preliminary-call reuse, cost-sensitive development-only threshold fitting, and difficulty/uncertainty/combined ablations.
- Accuracy, error, parse/timeout rates, latency distribution, input/output/total/reasoning tokens, throughput, peak accelerator memory, tokens per correct answer, verification rates, repeated-trial agreement/stability, bootstrap intervals, paired bootstrap comparisons, and exact McNemar tests.
- Resumable unique JSONL/CSV runs with model configuration and question-level call details.
- Twelve analysis figures when uncertainty data exist, including accuracy/error/latency/tokens/cost/difficulty/routing/verification/consistency/uncertainty/ablation/incremental-cost views; Pareto table; run-specific report.
- Streamlit Ask, task-aware Compare All, filtered Dashboard, and computed Research Insights. No mock results are presented as real.
- Unit/integration tests and a real small-model inference test. Real deterministic and stochastic smoke results are included.
- Complete README, research report, references, limitations, and presentation outline.

## Deliberately not claimed

- The generated-token uncertainty signal is not a calibrated correctness probability. Brier/ECE utilities and reliability-diagram support exist, but no probability calibration score is reported without valid probability inputs.
- Token counts are not labeled as energy or monetary cost. No precise GPU-energy result is claimed without external power telemetry.
- The synchronous Transformers timeout is a measured soft timeout; it cannot safely pre-empt an in-progress kernel.
- The same-model verifier is not independent.

## Pending full-scale evidence

- The full held-out `Qwen/Qwen3-4B` evaluation is configured but not executed locally. The 8.04 GB download was impractical on the available connection, while the 1.7B smoke model completed successfully.
- Smoke-study findings are not a substitute for the full final evaluation and must not be presented as definitive hypothesis confirmation.
