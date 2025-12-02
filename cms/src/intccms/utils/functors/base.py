"""Base functor executor for OO-based functor execution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import awkward as ak

from intccms.utils.functors.utils import get_function_arguments

class FunctorExecutor(ABC):
    """Abstract base class for functor executors.

    Execution pipeline: prepare_arguments → call function → apply_result

    Subclasses must implement apply_result() to define how the function
    result is used (e.g., applying mask, filling histogram, attaching field).
    """

    def __init__(self, config: Any):
        """Initialize with functor configuration.

        Parameters
        ----------
        config : FunctorConfig
            Configuration with function, use spec, and static_kwargs
        """
        self.config = config

    def prepare_arguments(
        self, objects: Dict[str, ak.Array]
    ) -> Tuple[List[ak.Array], Dict[str, Any]]:
        """Extract arguments from objects dict using functor's 'use' spec.

        Parameters
        ----------
        objects : Dict[str, ak.Array]
            Dictionary of awkward arrays

        Returns
        -------
        Tuple[List[ak.Array], Dict[str, Any]]
            Positional arguments and keyword arguments
        """
        return get_function_arguments(
            self.config.use,
            objects,
            function_name=self.config.function.__name__,
            static_kwargs=getattr(self.config, "static_kwargs", None),
        )

    @abstractmethod
    def apply_result(
        self, result: Any, objects: Dict[str, ak.Array], **kwargs
    ) -> Any:
        """Apply the function result.

        Must be implemented by subclasses. Use **kwargs to receive
        executor-specific parameters.

        Parameters
        ----------
        result : Any
            Result from functor function execution
        objects : Dict[str, ak.Array]
            Dictionary of awkward arrays (may be modified in-place)
        **kwargs
            Executor-specific parameters (e.g., object_name, field_name)

        Returns
        -------
        Any
            Subclass-specific return value
        """
        pass

    def execute(self, objects: Dict[str, ak.Array], **kwargs) -> Any:
        """Execute complete functor pipeline: prepare → call → apply.

        Parameters
        ----------
        objects : Dict[str, ak.Array]
            Dictionary of awkward arrays
        **kwargs
            Passed to apply_result() for executor-specific parameters

        Returns
        -------
        Any
            Result from apply_result()
        """
        args, static_kwargs = self.prepare_arguments(objects)
        result = self.config.function(*args, **static_kwargs)
        return self.apply_result(result, objects, **kwargs)
