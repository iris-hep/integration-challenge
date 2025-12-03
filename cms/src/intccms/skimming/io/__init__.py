"""I/O utilities for reading and writing event data.

This module provides readers and writers for various file formats used in
HEP analysis, with a focus on ROOT and Parquet formats compatible with
the NanoAOD schema.
"""

from .readers import RootReader, ParquetReader, get_reader
from .writers import RootWriter, ParquetWriter, get_writer
from .protocols import EventReader, EventWriter

__all__ = [
    "RootReader",
    "ParquetReader",
    "get_reader",
    "RootWriter",
    "ParquetWriter",
    "get_writer",
    "EventReader",
    "EventWriter",
]
