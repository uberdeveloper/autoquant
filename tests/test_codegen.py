# tests/test_codegen.py
"""Tests for codegen.py — smoke testing generated strategy modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import codegen


class TestSmokeFrame:
    def test_shape_and_columns(self):
        df = codegen.smoke_frame()
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 30
        assert df.index.is_monotonic_increasing


class TestSmokeTest:
    def write_module(self, tmp_path, source, name="m"):
        path = tmp_path / "strategies"
        path.mkdir(exist_ok=True)
        f = path / f"{name}.py"
        f.write_text(source)
        return f

    def test_valid_module_passes(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return (df.close > df.close.rolling(5).mean()).astype(float)\n"
        ))
        assert codegen.smoke_test(f, {}) is None

    def test_missing_signal_fails(self, tmp_path):
        f = self.write_module(tmp_path, "x = 1\n")
        err = codegen.smoke_test(f, {})
        assert err is not None and "signal" in err

    def test_crashing_signal_fails_with_exception(self, tmp_path):
        f = self.write_module(tmp_path, (
            "def signal(df, **params):\n"
            "    raise ValueError('boom')\n"
        ))
        err = codegen.smoke_test(f, {})
        assert err is not None and "ValueError" in err

    def test_wrong_index_fails(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return pd.Series([1.0])\n"
        ))
        err = codegen.smoke_test(f, {})
        assert err is not None and "index" in err

    def test_nan_output_fails(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def signal(df, **params):\n"
            "    return pd.Series(np.nan, index=df.index)\n"
        ))
        err = codegen.smoke_test(f, {})
        assert err is not None and "non-finite" in err

    def test_non_series_output_fails(self, tmp_path):
        f = self.write_module(tmp_path, "def signal(df, **params):\n    return 1.0\n")
        err = codegen.smoke_test(f, {})
        assert err is not None and "Series" in err

    def test_params_are_forwarded(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "def signal(df, lookback=5):\n"
            "    return (df.close > df.close.rolling(lookback).mean()).astype(float)\n"
        ))
        assert codegen.smoke_test(f, {"lookback": 10}) is None


class TestStripFence:
    def test_bare_source(self):
        assert codegen.strip_fence("x = 1") == "x = 1"

    def test_python_fence(self):
        assert codegen.strip_fence("```python\nx = 1\n```") == "x = 1"

    def test_bare_fence(self):
        assert codegen.strip_fence("```\nx = 1\n```") == "x = 1"


class TestBuildPrompt:
    def test_prompt_embeds_spec_and_contract(self):
        prompt = codegen.build_prompt({"meta": {"slug": "s"}, "signal": {"definition": "d"}})
        assert "def signal" in prompt
        assert "slug: s" in prompt
        assert "do NOT shift" in prompt
