"""Repack ROOT files: vendored upstream tool plus a Dask + XRootD layer.

- :mod:`root_repack` is vendored verbatim from
  https://github.com/pfackeldey/root_repack and provides the core
  plan/stage/merge machinery.
- :mod:`distributed` wraps that machinery for Dask workers that read
  inputs from XRootD and upload outputs back through ``xrdcp`` +
  ``xrdfs mv``.
"""

import sys

import cloudpickle

from intccms.utils.repack import distributed, root_repack
from intccms.utils.repack.distributed import (
    ChunkPlan,
    ChunkSegment,
    FileRecord,
    load_fileset,
    plan_chunks,
    prepare_output_dirs,
    repack_chunk,
    run_repack,
    write_output_fileset,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(root_repack)
cloudpickle.register_pickle_by_value(distributed)

__all__ = [
    "ChunkPlan",
    "ChunkSegment",
    "FileRecord",
    "distributed",
    "load_fileset",
    "plan_chunks",
    "prepare_output_dirs",
    "repack_chunk",
    "root_repack",
    "run_repack",
    "write_output_fileset",
]
