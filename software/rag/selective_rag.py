"""Compatibility wrapper for the implemented SelectiveRAG class.
The earlier duplicate module imported absent packages and was not executable.
"""
from . import SelectiveRAG  # noqa: F401
