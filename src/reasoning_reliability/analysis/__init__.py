from .reporting import computed_insights

__all__ = ["computed_insights", "generate_plots"]


def __getattr__(name: str):
    if name == "generate_plots":
        from .plots import generate_plots

        return generate_plots
    raise AttributeError(name)
