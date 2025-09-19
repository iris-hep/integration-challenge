import logging

from . import base as base
from . import diff as diff
from . import nondiff as nondiff


__all__ = [
    "base",
    "diff",
    "nondiff",
]


def __dir__():
    return __all__


def set_logging() -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s: %(name)s] %(message)s"
    )
