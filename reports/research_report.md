# Does Chain-of-Thought Actually Improve Decision Reliability?

## An Empirical Comparison of Direct, Verified, and Adaptive LLM Reasoning

### Abstract

This project provides a controlled, reproducible experiment for measuring whether explicit chain-of-thought (CoT) prompting improves large-language-model decision accuracy and reliability enough to justify its additional inference cost. One fixed open-weight model is evaluated on identical GSM8K, ARC-Easy, and ARC-Challenge questions using direct answering, prompt-based CoT, CoT followed by a separate verification call, and adaptive routing. A real Qwen3-1.7B deterministic smoke study (15 questions, 90 strategy/ablation records) and a separate three-trial stochastic reliability study (36 records) were completed. Verification achieved the highest smoke accuracy (73.3%) but used 4.85× the direct method's tokens and 25.17× its latency. These samples are too small for definitive hypothesis claims; the full held-out Qwen3-4B evaluation remains pending.

### 1. Introduction

CoT prompting can elicit useful intermediate computation, but more text and more inference calls do not automatically imply more correct or dependable decisions. A system may reason fluently toward a wrong answer, repeatedly reproduce the same error, or let a verifier replace a correct response with an incorrect one. The relevant engineering question is therefore not simply whether CoT sometimes helps, but when it helps, how much it costs, and whether a router can allocate that cost selectively.

### 2. Research question

Does CoT improve the accuracy and reliability of a fixed language model, and are any gains worth the extra latency and token consumption?

### 3. Hypotheses

- **H1:** CoT improves accuracy on sufficiently complex multi-step questions relative to direct answering.
- **H2:** Verification corrects some initial errors but also introduces harmful corrections and additional cost.
- **H3:** Adaptive reasoning produces a more favorable accuracy–cost trade-off than always applying expensive reasoning.

These are falsifiable hypotheses, not expected findings.

### 4. Related work

Wei et al. introduced CoT prompting as a means of eliciting intermediate reasoning in sufficiently large language models [1]. Wang et al. proposed self-consistency over multiple sampled reasoning paths [2], emphasizing that stochastic agreement is a method rather than proof of correctness. Cobbe et al. introduced GSM8K and studied learned verification for mathematical reasoning [3]. Clark et al. introduced ARC as a challenging science-question benchmark [4]. Later work examined process supervision [5] and test-time compute scaling [6]. The present experiment narrows the intervention: it fixes one model and compares workflow-level prompting, same-model verification, and interpretable routing.

### 5. Model and dataset selection

The primary configuration uses `Qwen/Qwen3-4B`, with `Qwen/Qwen3-1.7B` as the explicitly configured interactive/smoke alternative for constrained local hardware. Qwen3 supports native thinking and non-thinking chat-template modes [7]. The primary experiment disables native thinking and varies only prompts/workflow, so “prompt-based CoT” is not silently conflated with a model-native mode. Any native-thinking experiment must have a separate run ID and configuration label.

GSM8K provides multi-step grade-school mathematics. ARC-Easy and ARC-Challenge provide multiple-choice science and commonsense questions. Reference parsing accepts finite signed integers and decimals with common separators/currency characters. ARC evaluation compares normalized option keys, not unrestricted text.

### 6. Methodology

The official test records are deterministically ordered by a seeded hash and divided into disjoint development and evaluation pools. Exact selected examples are cached, duplicate IDs are rejected, and all strategies receive the same records. Prompts, router thresholds, and cost weights may be selected only on development data.

The primary accuracy comparison is deterministic where model guidance permits. Stochastic repeated trials are a separate experiment with at least three trials per question. Question-level—not generation-level—resampling is used for paired inference.

### 7. Reasoning approaches

**Direct answering** requests only a delimited final answer and performs one inference call. **Prompt-based CoT** requests concise intermediate steps followed by the same delimiter and also performs one call. **CoT with verification** first produces CoT, then passes the original question, proposed reasoning, and proposed answer—but never the reference—to a separate verifier call. A robust parser retains the initial answer if the verifier does not supply a valid correction.

Verification is categorized after scoring as wrong→right, right→wrong, right preserved, or wrong preserved. Because solver and verifier use the same model weights, verification is not statistically or epistemically independent.

### 8. Adaptive reasoning architecture

An interpretable difficulty estimator combines length, sentence count, numeric quantities, operation terms, constraint terms, reasoning indicators, task type, and option count. A preliminary direct response supplies an uncertainty proxy from mean generated-token likelihood when token scores exist. Missing scores produce an explicit fallback rather than synthetic confidence.

The routing score is a configurable convex combination of difficulty and uncertainty. Scores below the low threshold reuse the preliminary direct response; moderate scores invoke CoT; high scores invoke verified CoT. All preliminary, routing, and subsequent inference costs count toward adaptive totals. Ablations use difficulty only, uncertainty only, and the combined signal. The cost-sensitive development objective is `accuracy - lambda * normalized_cost`, where cost must be normalized within a compatible development run.

### 9. Experimental setup

Every run records model ID/revision, precision, quantization, hardware, PyTorch version, generation settings, random seed, strategy calls, selected route, failures, and question-level measurements. Model loading and warm-up are excluded from timed question inference. CUDA or Apple MPS synchronization surrounds accelerated generation, and allocated accelerator memory is recorded. The model is loaded once and shared.

Natural strategy budgets measure realistic resource use. A separately labeled matched-budget analysis may cap output lengths, but it must not replace the natural-budget comparison. Runs with different models, native-thinking settings, precision, or quantization are not merged.

### 10. Evaluation metrics

Primary accuracy treats invalid, missing, and timed-out predictions as wrong. Separate rates describe parsing and timeout failures. Latency includes preprocessing, routing, preliminary prediction, reasoning, and verification. Token reports separate input and generated tokens and compute tokens per correct answer. Peak allocated GPU memory and throughput are measured; token counts are not represented as energy or financial cost.

Reliability includes per-question answer agreement, correctness stability, and accuracy variance across trials. Consistency is not correctness. For probabilistic confidence that is defensibly normalized, Brier score, expected calibration error, and reliability diagrams are appropriate. The current open-ended likelihood signal is only an uncertainty proxy, so probability-calibration claims are withheld.

Accuracy receives a question-level bootstrap 95% confidence interval. Identical-question strategy comparisons use paired bootstrap differences and exact McNemar tests. Small samples are flagged as exploratory.

### 11. Results

The deterministic development smoke run used five questions from each dataset and Qwen3-1.7B in non-thinking FP16 mode on Apple MPS. Direct answering achieved 53.3% accuracy, CoT 66.7%, verification 73.3%, and combined adaptive routing 53.3%. Difficulty-only adaptive routing achieved 66.7%; uncertainty-only routing matched direct at 53.3%. Verification preserved ten initially correct answers and preserved four wrong answers among the 14 parseable initial solutions; it recorded no harmful correction and no successful wrong-to-right correction in this small run. One initial CoT response was unparseable.

| Strategy | Accuracy | Mean latency (s) | Mean total tokens | Relative latency vs direct | Relative tokens vs direct |
|---|---:|---:|---:|---:|---:|
| Direct | 53.3% | 0.563 | 128.3 | 1.00× | 1.00× |
| CoT | 66.7% | 6.820 | 250.1 | 12.11× | 1.95× |
| Verification | 73.3% | 14.176 | 622.4 | 25.17× | 4.85× |
| Adaptive combined | 53.3% | 0.547 | 128.3 | 0.97× | 1.00× |
| Adaptive difficulty-only | 66.7% | 1.768 | 177.1 | 3.14× | 1.38× |
| Adaptive uncertainty-only | 53.3% | 0.534 | 128.3 | 0.95× | 1.00× |

The development-only cost-sensitive threshold search selected low/high thresholds of 0.1/0.9 with measured development utility 0.723 for λ=0.1. This fitted result is analytical development output and was not presented as held-out performance.

The separate stochastic reliability run used one question per dataset, three trials, and all four primary strategies. Direct and combined adaptive accuracy were 66.7% with answer agreement 1.0; verification accuracy was 44.4%, agreement 0.778, and correctness stability 0.778. CoT reached the 96-token reliability cap before a valid final delimiter on all sampled records, producing 0% primary accuracy; this is a matched configuration failure and demonstrates why consistency and correctness must remain separate.

### 12. Statistical analysis

Question-level paired bootstrap intervals and exact McNemar tests were generated for all deterministic strategy pairs. For direct versus verification, the measured accuracy difference was −0.20 (95% bootstrap interval −0.467 to 0.067; exact McNemar p=0.375). No fixed-strategy difference was statistically decisive at this sample size. Repeated generations in the reliability study were summarized within question and were not treated as independent questions.

### 13. Discussion

The smoke evidence is directionally consistent with H1 on this sample: CoT improved accuracy over direct by 13.3 percentage points, but its paired confidence interval included zero and latency increased roughly twelvefold. Verification added another 6.7 points over CoT while substantially increasing cost; it did not actually correct a parseable wrong answer in the deterministic sample, so the gain partly reflects recovery around parsing/response variation rather than proven independent checking. The combined adaptive rule routed almost entirely to direct and therefore reproduced direct accuracy and cost. Difficulty-only routing offered a more favorable observed compromise, but this is development data and cannot establish H3. The stochastic study further shows that a method may be perfectly consistent while consistently invalid or wrong.

### 14. Limitations and threats to validity

- Results from one model family may not generalize to other architectures or scales.
- Same-model verification can reproduce correlated errors.
- Prompt wording is an intervention and may interact with task type.
- Generated-token likelihood is not a calibrated correctness probability, especially for GSM8K.
- Benchmark contamination cannot be ruled out from open model training data.
- ARC/GSM8K represent limited task domains and simplified answer formats.
- Software token counts and peak allocated memory are incomplete proxies for hardware energy.
- The synchronous inference timeout is a soft threshold rather than pre-emption.
- Small development sets may overfit routing thresholds.

### 15. Conclusion

The completed Qwen3-1.7B smoke studies show that more reasoning can improve measured accuracy while imposing large latency and token costs, and that verification is not automatically corrective. They do not establish the hypotheses because the samples are small and development-oriented. The defensible conclusion is therefore provisional: CoT and verification were more accurate in the deterministic smoke sample, direct/adaptive-direct were far more efficient, and the routing policy requires frozen-threshold validation on the pending full Qwen3-4B held-out evaluation.

### 16. Future work

Future studies should compare independently trained verifiers, calibrated choice likelihoods, learned routers with nested validation, process-supervision signals, direct hardware energy telemetry, additional model families and sizes, adversarial consistency tests, and preregistered final configurations.

### 17. References

1. Jason Wei et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” *NeurIPS 2022*. https://arxiv.org/abs/2201.11903
2. Xuezhi Wang et al. “Self-Consistency Improves Chain of Thought Reasoning in Language Models.” *ICLR 2023*. https://arxiv.org/abs/2203.11171
3. Karl Cobbe et al. “Training Verifiers to Solve Math Word Problems.” 2021. https://arxiv.org/abs/2110.14168
4. Peter Clark et al. “Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.” 2018. https://arxiv.org/abs/1803.05457
5. Hunter Lightman et al. “Let’s Verify Step by Step.” 2023. https://arxiv.org/abs/2305.20050
6. Charlie Snell et al. “Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.” 2024. https://arxiv.org/abs/2408.03314
7. Qwen Team. “Qwen3-4B Model Card.” https://huggingface.co/Qwen/Qwen3-4B
8. Hugging Face. “Transformers Text Generation.” https://huggingface.co/docs/transformers/main_classes/text_generation
9. OpenAI. “GSM8K Dataset Card.” https://huggingface.co/datasets/openai/gsm8k
10. Allen Institute for AI. “AI2 ARC Dataset Card.” https://huggingface.co/datasets/allenai/ai2_arc

