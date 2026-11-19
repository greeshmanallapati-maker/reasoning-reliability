# Reasoning Reliability

Research code for **“Does Chain-of-Thought Actually Improve Decision Reliability?”** It compares direct answering, prompt-based chain-of-thought (CoT), CoT plus a separate verification call, and cost-aware adaptive routing on GSM8K and AI2 ARC using identical model weights and identical selected questions.

The repository contains no fabricated benchmark data. Real deterministic smoke and stochastic reliability runs are stored under `results/`; the full Qwen3-4B evaluation remains pending.

## Research questions

- Does explicit prompt-based CoT improve accuracy enough to justify its latency and token cost?
- Does a second verification call correct more errors than it introduces?
- Can interpretable difficulty and model-likelihood uncertainty signals route questions more efficiently?

The hypotheses are tested rather than assumed. A repeated wrong answer counts as consistent, not correct or reliable.

## Architecture

```text
question -> deterministic benchmark record -> shared model engine
                                             |-> direct (1 call)
                                             |-> CoT (1 call)
                                             |-> verified CoT (2 calls)
                                             `-> adaptive (preliminary direct call, then reuse or route)
                                                       |
                                                       -> result store -> metrics -> figures/report
```

`Qwen/Qwen3-4B` is the primary default. All four strategies share one loaded model. The primary experiment sets Qwen3 native thinking **off**, then changes only the prompt/workflow. This isolates prompt-based CoT from Qwen3's model-native `enable_thinking` mode. A secondary native-thinking experiment can be run as a separately labeled configuration and must never be merged silently with the primary run.

## Project layout

```text
configs/                 reproducible smoke/default/full YAML configurations
src/reasoning_reliability/
  model/                 shared HF model loader and timed inference
  data/                  GSM8K/ARC loading, normalization, cached sampling
  strategies/            direct, CoT, verification, adaptive workflows
  adaptive/              difficulty, uncertainty, routing policy
  evaluation/            benchmark, storage, metrics, paired statistics
  analysis/              publication plots and computed insights
app/streamlit_app.py      interactive research demonstration
scripts/                  benchmark, analysis, and smoke entry points
tests/                    unit, integration, resume, accounting tests
reports/                  report and presentation outline
results/                  unique run directories (ignored by Git)
```

## Installation

Python 3.10+ is required. CUDA is recommended for the 4B model. Apple Silicon Macs use the PyTorch MPS backend automatically; CPU remains the final fallback.

```bash
git clone https://github.com/greeshmanallapati-maker/reasoning-reliability.git
cd reasoning-reliability
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Optional 4-bit support:

```bash
python -m pip install -e '.[quantization,test]'
```

Check PyTorch acceleration:

```bash
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False})"
```

The first benchmark invocation downloads the configured model and datasets from Hugging Face. The Streamlit demo defaults to the stronger Qwen3-4B model. To prefetch it explicitly:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download Qwen/Qwen3-4B
```

The supplied configurations set `model.hub_auth: anonymous`. This prevents an expired
cached Hugging Face login from being attached to requests for the public Qwen repositories.
For a private or gated model, run `hf auth login` and change the setting to
`hub_auth: authenticated`.

The Streamlit `demo.yaml` configuration sets `local_files_only: true` after Qwen3-4B
has been downloaded. This makes the demo independent of Hub login state. On a lower-memory
machine, enter `configs/demo_light.yaml` in the Configuration field to use Qwen3-1.7B.
Set `local_files_only` to `false` only when intentionally refreshing model files.

## Run the experiments

Smoke benchmark (5 examples from each of GSM8K, ARC-Easy, and ARC-Challenge; Qwen3-1.7B):

```bash
python scripts/run_benchmark.py --config configs/smoke.yaml
```

Development benchmark:

```bash
python scripts/run_benchmark.py --config configs/default.yaml
```

Full benchmark:

```bash
python scripts/run_benchmark.py --config configs/full.yaml --ablations
```

Resume an interrupted run without duplicating completed `(question, strategy, trial)` records:

```bash
python scripts/run_benchmark.py --config configs/default.yaml --run-id YOUR_RUN_ID --resume
```

Generate summary tables, Pareto labels, computed insights, and figures:

```bash
python scripts/run_analysis.py results/YOUR_RUN_ID
```

Fit adaptive thresholds from a development run only:

```bash
python scripts/fit_router.py results/YOUR_DEVELOPMENT_RUN_ID
```

Run three-trial stochastic reliability and the separately labeled native-thinking study:

```bash
python scripts/run_benchmark.py --config configs/reliability.yaml
python scripts/run_benchmark.py --config configs/native_thinking_smoke.yaml
```

Launch the UI:

```bash
cd reasoning-reliability
streamlit run app/streamlit_app.py --server.port 8501
```

Keep that terminal open and use the single URL it prints. The Compare All workflow reuses
the direct, CoT, and verification generations for adaptive routing, avoiding redundant
Apple-MPS work while retaining the logical cost accounting for each strategy.

### Deploy on Streamlit Community Cloud

The app automatically defaults to `configs/demo_cloud.yaml` on Linux hosting. This profile
uses Qwen3-0.6B on CPU so the public demo can run without a GPU; macOS continues to default
to the local Qwen3-4B profile. The entrypoint-specific `app/requirements.txt` keeps the cloud
environment smaller than the full research installation.

At [share.streamlit.io](https://share.streamlit.io/), create an app with:

- Repository: `greeshmanallapati-maker/reasoning-reliability`
- Branch: `main`
- Entrypoint: `app/streamlit_app.py`
- Python: `3.12`

No Hugging Face token is required because the cloud profile downloads a public model using
anonymous access. The first model-backed request can take longer while the weights are cached.

Run tests without downloading model weights:

```bash
pytest -q
```

Enable the opt-in real inference integration test:

```bash
RUN_REAL_MODEL_TEST=1 pytest -q -m integration
```

## Data controls

The loader uses official Hugging Face datasets `openai/gsm8k` (`main`) and `allenai/ai2_arc` (`ARC-Easy` and `ARC-Challenge`). It creates deterministic, disjoint development/evaluation pools from the official test split, rejects duplicate IDs, caches exact selected records, and passes identical examples to every strategy. Development answers may be used to choose thresholds; evaluation answers must not be used for tuning.

## Metrics

- **Accuracy/error:** invalid, missing, and timed-out outputs fail primary accuracy.
- **Latency:** wall-clock end-to-end strategy latency includes routing and all calls; model-call latency is also stored.
- **Tokens:** input, generated, and total tokens are separate; tokens per correct answer is reported.
- **Compute:** peak allocated GPU memory, throughput, and token counts are measured. Tokens are a compute proxy, not an energy measurement.
- **Verification:** wrong→right, right→wrong, right preserved, and wrong preserved are reported.
- **Reliability:** repeated-run agreement and correctness are reported separately.
- **Statistics:** question-level bootstrap confidence intervals, paired bootstrap differences, and exact McNemar tests are available. Small samples must be interpreted cautiously.
- **Calibration:** generated-token likelihood is labeled an uncertainty proxy, not a calibrated probability. ECE/Brier scores must only be used if a later experiment supplies defensible probabilities.

## Fairness and known limitations

- The primary deterministic experiment keeps model weights, precision, native-thinking mode, sampling, datasets, and scoring fixed.
- `benchmark.batch_size` controls real padded batching for compatible one-call direct/CoT workflows. Verification and adaptive workflows remain sequential because later calls depend on earlier outputs.
- Natural strategy budgets are reported alongside token cost; CoT is not forced to use direct answering's output length.
- Qwen3 native thinking and prompt-based CoT are separate variables. The engine passes `enable_thinking` through the model chat template for Qwen3.
- A verifier using the same model is not independent and can reinforce the original error.
- Mean generated-token likelihood is weak for open-ended numeric answers and is not calibrated confidence.
- A timeout is currently a soft post-generation limit because synchronous Transformers generation is not safely pre-empted. It remains an auditable failure.
- FLOPs and energy are intentionally not reported as measured values. Add hardware telemetry before making energy claims.
- Default adaptive thresholds are transparent initial rules. Fit/select them only on development data before final evaluation.

## Reproducing a defensible study

1. Record the GPU, package environment, model revision, precision, and quantization in the generated metadata.
2. Tune prompts and routing thresholds only on the development partition.
3. Freeze configuration and run the evaluation partition once.
4. Run stochastic consistency trials in a separate configuration (`do_sample: true`, `trials: 3`) rather than mixing them into deterministic accuracy results.
5. Generate figures and update the pending Results/Conclusion sections of the report only from the resulting CSV/JSONL files.

## Included real runs

- `results/smoke-real-20261118`: 15 questions × 6 fixed/adaptive-ablation workflows = 90 deterministic records.
- `results/reliability-real-compact-20261118`: 3 questions × 4 strategies × 3 stochastic trials = 36 reliability records.

These are small engineering/research smoke studies, not substitutes for the full held-out Qwen3-4B evaluation.

## Primary sources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Hugging Face generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [GSM8K paper](https://arxiv.org/abs/2110.14168) and [dataset](https://huggingface.co/datasets/openai/gsm8k)
- [AI2 ARC paper](https://arxiv.org/abs/1803.05457) and [dataset](https://huggingface.co/datasets/allenai/ai2_arc)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

