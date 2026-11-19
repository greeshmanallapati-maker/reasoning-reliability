from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol


def configure_hub_auth(mode: str) -> None:
    """Configure documented Hub implicit-token behavior before Hub imports."""
    if mode == "anonymous":
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    elif mode == "authenticated":
        os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
    else:
        raise ValueError("model.hub_auth must be 'anonymous' or 'authenticated'")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


@dataclass
class GenerationResult:
    prompt: str
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    model_id: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    token_logprobs: list[float] = field(default_factory=list)
    peak_gpu_memory_bytes: int = 0
    error: str | None = None
    timed_out: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def mean_logprob(self) -> float | None:
        return sum(self.token_logprobs) / len(self.token_logprobs) if self.token_logprobs else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"total_tokens": self.total_tokens, "mean_logprob": self.mean_logprob}


class InferenceEngine(Protocol):
    model_id: str

    def generate(self, prompt: str, **overrides: Any) -> GenerationResult: ...

    def generate_batch(self, prompts: list[str], **overrides: Any) -> list[GenerationResult]: ...

    def count_tokens(self, text: str) -> int: ...


class HuggingFaceEngine:
    """One shared model instance for all reasoning strategies."""

    def __init__(self, model_config: dict[str, Any], generation_config: dict[str, Any]) -> None:
        # huggingface_hub otherwise sends a cached login token even for public
        # repositories. An expired ambient token can therefore break public model
        # downloads with a misleading RepositoryNotFoundError. This environment
        # variable is the Hub's documented opt-out and must be set before importing
        # Transformers / huggingface_hub.
        self.hub_auth = model_config.get("hub_auth", "anonymous")
        configure_hub_auth(self.hub_auth)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, __version__ as transformers_version

        self.torch = torch
        self.model_id = model_config["model_id"]
        self.revision = model_config.get("revision", "main")
        self.native_thinking = bool(model_config.get("native_thinking", False))
        self.local_files_only = bool(model_config.get("local_files_only", False))
        self.generation_config = dict(generation_config)
        self.timeout_seconds = float(self.generation_config.pop("timeout_seconds", 180))
        device = model_config.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        precision = model_config.get("precision", "auto")
        dtype = self._resolve_dtype(precision)
        dtype_argument = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            dtype_argument: dtype,
        }
        tokenizer_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
        }
        if self.hub_auth == "anonymous":
            # Explicitly override an already-imported Hub client as well. This is
            # important for Streamlit hot reloads after an earlier failed login.
            kwargs["token"] = False
            tokenizer_kwargs["token"] = False
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        quantization = model_config.get("quantization", "none")
        if quantization == "4bit":
            if self.device != "cuda":
                raise ValueError("4-bit bitsandbytes quantization requires CUDA")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device in {"cpu", "mps"}:
            self.model.to(self.device)
        self.model.eval()
        self.load_metadata = {
            "model_id": self.model_id,
            "revision": self.revision,
            "precision": str(dtype),
            "quantization": quantization,
            "hub_auth": self.hub_auth,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "hardware": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "Apple Metal (MPS)"
                if self.device == "mps"
                else platform.processor()
            ),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "transformers": transformers_version,
            "datasets": _package_version("datasets"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
        }

    def _resolve_dtype(self, precision: str) -> Any:
        torch = self.torch
        if precision == "float32" or self.device == "cpu":
            return torch.float32
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16
        if self.device == "mps":
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def _format_prompt(self, prompt: str) -> Any:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        # Qwen3 exposes native thinking separately from prompt-based CoT.
        if "qwen3" in self.model_id.lower():
            kwargs["enable_thinking"] = self.native_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def warmup(self) -> None:
        self.generate("Reply with FINAL_ANSWER: A", max_new_tokens=4, collect_token_scores=False)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str, **overrides: Any) -> GenerationResult:
        torch = self.torch
        settings = self.generation_config | overrides
        collect_token_scores = bool(settings.pop("collect_token_scores", True))
        if not settings.get("do_sample", False):
            settings.pop("temperature", None)
            settings.pop("top_p", None)
        result = GenerationResult(prompt=prompt, model_id=self.model_id, settings=settings)
        started = time.perf_counter()
        try:
            encoded = self._format_prompt(prompt)
            if hasattr(encoded, "keys"):
                encoded = encoded.to(self.model.device)
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
            else:
                input_ids = encoded.to(self.model.device)
                attention_mask = torch.ones_like(input_ids)
            result.input_tokens = int(input_ids.shape[-1])
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=collect_token_scores,
                    **settings,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
                result.peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
            elif self.device == "mps":
                torch.mps.synchronize()
                result.peak_gpu_memory_bytes = int(torch.mps.current_allocated_memory())
            result.latency_seconds = time.perf_counter() - started
            generated = output.sequences[0, input_ids.shape[-1] :]
            result.output_tokens = int(generated.shape[-1])
            result.text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            if collect_token_scores:
                for token_id, logits in zip(generated, output.scores, strict=False):
                    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
                    result.token_logprobs.append(float(log_probs[token_id].item()))
            result.timed_out = result.latency_seconds > self.timeout_seconds
            if result.timed_out:
                result.error = f"Generation exceeded soft timeout ({self.timeout_seconds}s)"
            del output, encoded
            if self.device == "mps":
                torch.mps.empty_cache()
        except torch.cuda.OutOfMemoryError as exc:
            result.error = f"CUDA out of memory: {exc}"
            result.latency_seconds = time.perf_counter() - started
            torch.cuda.empty_cache()
        except Exception as exc:  # benchmark failures must become auditable records
            result.error = f"{type(exc).__name__}: {exc}"
            result.latency_seconds = time.perf_counter() - started
        return result

    def generate_batch(self, prompts: list[str], **overrides: Any) -> list[GenerationResult]:
        """Generate a padded batch for compatible one-call strategies."""
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], **overrides)]
        torch = self.torch
        settings = self.generation_config | overrides
        collect_token_scores = bool(settings.pop("collect_token_scores", True))
        if not settings.get("do_sample", False):
            settings.pop("temperature", None)
            settings.pop("top_p", None)
        results = [GenerationResult(prompt=p, model_id=self.model_id, settings=settings) for p in prompts]
        started = time.perf_counter()
        try:
            template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if "qwen3" in self.model_id.lower():
                template_kwargs["enable_thinking"] = self.native_thinking
            rendered = [
                self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **template_kwargs)
                for prompt in prompts
            ]
            previous_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(rendered, padding=True, return_tensors="pt").to(self.model.device)
            self.tokenizer.padding_side = previous_padding_side
            input_width = int(encoded["input_ids"].shape[-1])
            input_counts = encoded["attention_mask"].sum(dim=1).tolist()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=collect_token_scores,
                    **settings,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            elif self.device == "mps":
                torch.mps.synchronize()
                peak = int(torch.mps.current_allocated_memory())
            else:
                peak = 0
            elapsed_per_item = (time.perf_counter() - started) / len(prompts)
            generated = output.sequences[:, input_width:]
            for index, result in enumerate(results):
                token_ids = generated[index].tolist()
                while token_ids and token_ids[-1] == self.tokenizer.pad_token_id:
                    token_ids.pop()
                result.input_tokens = int(input_counts[index])
                result.output_tokens = len(token_ids)
                result.text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                result.latency_seconds = elapsed_per_item
                result.peak_gpu_memory_bytes = peak
                if collect_token_scores:
                    for step, token_id in enumerate(token_ids):
                        if step >= len(output.scores):
                            break
                        log_probs = torch.log_softmax(output.scores[step][index].float(), dim=-1)
                        result.token_logprobs.append(float(log_probs[token_id].item()))
                result.timed_out = result.latency_seconds > self.timeout_seconds
                if result.timed_out:
                    result.error = f"Generation exceeded soft timeout ({self.timeout_seconds}s)"
            del output, encoded
            if self.device == "mps":
                torch.mps.empty_cache()
        except Exception as exc:
            elapsed = (time.perf_counter() - started) / len(prompts)
            for result in results:
                result.error = f"{type(exc).__name__}: {exc}"
                result.latency_seconds = elapsed
            if self.device == "cuda" and isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        return results

