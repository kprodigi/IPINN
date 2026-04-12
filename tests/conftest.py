"""Shared fixtures for IPINN test suite."""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MOD_NAME = "composite_design_v19_test"


@pytest.fixture(scope="session")
def m():
    """Load the main module once per test session."""
    path = os.path.join(ROOT, "composite_design_v19.py")
    spec = importlib.util.spec_from_file_location(_MOD_NAME, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        sys.modules.pop(_MOD_NAME, None)


@pytest.fixture()
def tiny_df():
    """Load the tiny_crush.csv fixture as a DataFrame."""
    csv_path = os.path.join(ROOT, "tests", "fixtures", "tiny_crush.csv")
    return pd.read_csv(csv_path)


@pytest.fixture()
def logger(tmp_path):
    """Create a temporary logger for tests."""
    import logging

    log = logging.getLogger(f"test_{id(tmp_path)}")
    log.setLevel(logging.DEBUG)
    log.handlers = []
    fh = logging.FileHandler(os.path.join(str(tmp_path), "test.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    return log
