"""Pure helper functions for dataset configuration processing."""

from pathlib import Path
from typing import Callable, List, Optional, TypeVar, Union

T = TypeVar("T")


def normalize_to_list(
    value: Union[T, tuple, list], transform: Optional[Callable[[T], T]] = None
) -> List[T]:
    """
    Normalize a single value or sequence to a list, optionally transforming elements.

    Parameters
    ----------
    value : Union[T, tuple, list]
        Single value, tuple, or list to normalize
    transform : Optional[Callable[[T], T]]
        Optional transform function to apply to each element (e.g., Path)

    Returns
    -------
    List[T]
        Normalized list with optional transformation applied

    Examples
    --------
    >>> normalize_to_list(3)
    [3]
    >>> normalize_to_list([1, 2, 3])
    [1, 2, 3]
    >>> normalize_to_list("path/to/dir", transform=Path)
    [Path("path/to/dir")]
    >>> normalize_to_list(["a", "b"], transform=Path)
    [Path("a"), Path("b")]
    """
    if isinstance(value, (tuple, list)):
        items = list(value)
    else:
        items = [value]

    if transform is not None:
        items = [transform(item) for item in items]

    return items


def replicate_single(items: List[T], target_length: int) -> List[T]:
    """
    Replicate single-item list to target length, or validate multi-item list length.

    If items contains a single value and target_length > 1, replicates that value
    to match target_length. If items already has multiple values, validates that
    the length matches target_length.

    Parameters
    ----------
    items : List[T]
        List to optionally replicate
    target_length : int
        Target length (typically number of directories)

    Returns
    -------
    List[T]
        Replicated list if single item, or original list if already correct length

    Raises
    ------
    ValueError
        If items has multiple values but length doesn't match target_length

    Examples
    --------
    >>> replicate_single([0.5], 3)
    [0.5, 0.5, 0.5]
    >>> replicate_single([0.1, 0.2, 0.3], 3)
    [0.1, 0.2, 0.3]
    >>> replicate_single([0.1, 0.2], 3)
    ValueError: Expected 3 items to match target length, got 2
    """
    if len(items) == 1 and target_length > 1:
        return items * target_length
    elif len(items) != target_length:
        raise ValueError(
            f"Expected {target_length} items to match target length, got {len(items)}"
        )
    return items


def index_or_scalar(
    value: Union[T, tuple, list], index: Optional[int] = None, context: str = "item"
) -> T:
    """
    Index into a sequence or return scalar value, with optional bounds checking.

    If value is a sequence and index is provided, returns the element at that index.
    If value is a sequence and index is None, returns the first element.
    If value is a scalar, returns it directly (ignoring index).

    Parameters
    ----------
    value : Union[T, tuple, list]
        Single value, tuple, or list
    index : Optional[int]
        Index to access (if value is a sequence). If None, returns first element
        or the scalar value
    context : str
        Context description for error messages (e.g., 'lumi_mask', 'process')

    Returns
    -------
    T
        Indexed or scalar value

    Raises
    ------
    ValueError
        If index is out of bounds for the sequence

    Examples
    --------
    >>> index_or_scalar([1, 2, 3], index=0)
    1
    >>> index_or_scalar([1, 2, 3], index=None)
    1
    >>> index_or_scalar(42, index=0)
    42
    >>> index_or_scalar([1, 2], index=5, context="test")
    ValueError: index 5 out of range for test with 2 items
    """
    if isinstance(value, (tuple, list)):
        if index is not None:
            if index < 0 or index >= len(value):
                raise ValueError(
                    f"index {index} out of range for {context} with {len(value)} items"
                )
            return value[index]
        return value[0]

    return value


def count_directories(directories: Union[str, tuple, list]) -> int:
    """
    Count the number of directories from a single path or sequence of paths.

    Parameters
    ----------
    directories : Union[str, tuple, list]
        Single directory path or sequence of directory paths

    Returns
    -------
    int
        Number of directories

    Examples
    --------
    >>> count_directories("path/to/dir")
    1
    >>> count_directories(["dir1", "dir2", "dir3"])
    3
    >>> count_directories(("dir1", "dir2"))
    2
    """
    if isinstance(directories, str):
        return 1
    return len(directories)
