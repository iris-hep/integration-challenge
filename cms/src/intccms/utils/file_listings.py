"""Read ROOT file path listings from .txt files."""

import logging
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)


def collect_file_paths(
    directory: Union[str, Path],
    identifiers: int | List[int] | None = None,
    redirector: str | None = None,
    skip_files: List[str] | None = None,
) -> List[str]:
    """
    Read ROOT file paths from .txt listing files.

    Reads .txt files where each line contains one ROOT file path. This approach
    separates file lists from code, enabling version control and easy updates.
    The `identifiers` parameter allows processing subsets for testing.
    The `redirector` parameter prepends protocol prefixes for remote access.

    Parameters
    ----------
    directory : str or Path
        Directory containing .txt listing files
    identifiers : int or list of ints, optional
        Process only specific listing files by ID (e.g., [0, 1] reads 0.txt, 1.txt).
        If None, reads all .txt files in directory.
    redirector : str, optional
        URL prefix to prepend to paths (e.g., "root://xrootd.server.com//").
        If None, paths used as-is.
    skip_files : list of str, optional
        File paths (or substrings) to skip. Any file whose path contains
        one of these strings will be excluded.

    Returns
    -------
    List[str]
        ROOT file paths with optional redirector prefix applied.

    Raises
    ------
    FileNotFoundError
        If directory contains no .txt files or specified identifier file missing.
    """
    dir_path = Path(directory)

    # Determine which text files to parse
    if identifiers is None:
        listing_files = list(dir_path.glob("*.txt"))
    else:
        ids = [identifiers] if isinstance(identifiers, int) else identifiers
        listing_files = [dir_path / f"{i}.txt" for i in ids]

    # Raise error if no listing files are found
    if not listing_files:
        raise FileNotFoundError(f"No listing files found in {dir_path}")

    skip = skip_files or []
    root_paths: List[str] = []

    # Iterate through each listing file
    for txt_file in listing_files:
        if not txt_file.is_file():
            raise FileNotFoundError(f"Missing listing file: {txt_file}")

        # Read each non-empty line as a file path
        for line in txt_file.read_text().splitlines():
            path_str = line.strip()
            if path_str:
                if redirector:
                    path_str = f"{redirector}{path_str}"
                if any(s in path_str for s in skip):
                    logger.debug(f"Skipping file: {path_str}")
                    continue
                root_paths.append(path_str)

    return root_paths
