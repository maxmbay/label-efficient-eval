"""Figure builders."""

__all__ = ["plot_m1"]


def __getattr__(name: str):
    if name == "plot_m1":
        from lee.figures.m1 import plot_m1

        return plot_m1
    raise AttributeError(name)
