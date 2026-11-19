from .metrics import pareto_frontier, summarize_results

__all__ = ["BenchmarkRunner", "pareto_frontier", "summarize_results"]


def __getattr__(name: str):
    if name == "BenchmarkRunner":
        from .benchmark import BenchmarkRunner

        return BenchmarkRunner
    raise AttributeError(name)
