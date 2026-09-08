# tests/test_codegen.py
"""Tests for codegen.py — smoke testing generated strategy modules."""
from __future__ import annotations

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
        # the module returns NaN unless the param equals the forwarded value,
        # so dropping the param makes the smoke fail
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def signal(df, lookback=5):\n"
            "    if lookback != 10: return pd.Series(np.nan, index=df.index)\n"
            "    return (df.close > df.close.rolling(lookback).mean()).astype(float)\n"
        ))
        assert codegen.smoke_test(f, {"lookback": 10}) is None
        assert codegen.smoke_test(f, {}) is not None


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


VALID_MODULE = (
    "import pandas as pd\n"
    "def signal(df, **params):\n"
    "    return (df.close > df.close.rolling(5).mean()).astype(float)\n"
)


def cli_stub(stdout, returncode=0):
    def fake_run(*a, **k):
        class Proc:
            pass
        Proc.returncode = returncode
        Proc.stdout = stdout
        Proc.stderr = ""
        return Proc()
    return fake_run


def write_spec(tmp_path, slug="test-slug"):
    p = tmp_path / "specs" / f"{slug}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("meta:\n"
                 f"  slug: {slug}\n"
                 "signal:\n"
                 "  definition: d\n"
                 "  lag_bars: 1\n")
    return p


class TestGenerateOne:
    def test_valid_reply_returns_source(self, tmp_path, monkeypatch):
        spec_path = write_spec(tmp_path)
        monkeypatch.setattr(codegen.subprocess, "run", cli_stub(VALID_MODULE))
        result = codegen.generate_one(spec_path, None)
        assert result["slug"] == "test-slug"
        assert "def signal" in result["source"]

    def test_cli_failure_returns_error(self, tmp_path, monkeypatch):
        spec_path = write_spec(tmp_path)
        monkeypatch.setattr(codegen.subprocess, "run", cli_stub("", returncode=1))
        result = codegen.generate_one(spec_path, None)
        assert result["slug"] == "test-slug"
        assert "error" in result

    def test_malformed_spec_returns_error_not_raise(self, tmp_path, monkeypatch):
        spec_path = tmp_path / "specs" / "bad-slug.yaml"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text("not: [valid")  # unclosed flow sequence
        monkeypatch.setattr(codegen.subprocess, "run", cli_stub(VALID_MODULE))
        result = codegen.generate_one(spec_path, None)
        assert "error" in result
        assert result["slug"] == "bad-slug"

    def test_slug_mismatch_returns_error(self, tmp_path, monkeypatch):
        spec_path = tmp_path / "specs" / "mis.yaml"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text("meta:\n  slug: other-name\n")
        monkeypatch.setattr(codegen.subprocess, "run", cli_stub(VALID_MODULE))
        result = codegen.generate_one(spec_path, None)
        assert "error" in result
        assert "match" in result["error"]


class TestPublish:
    def test_success_smoke_passes_and_publishes(self, tmp_path):
        spec = {"meta": {"slug": "test-slug"},
                "signal": {"definition": "d", "lag_bars": 1}}
        result = {"slug": "test-slug", "source": VALID_MODULE}
        stage = codegen.publish(result, spec, tmp_path / "strategies")
        assert stage == "coded"
        assert (tmp_path / "strategies" / "test-slug.py").exists()
        assert not (tmp_path / "strategies" / "test-slug.py.tmp").exists()

    def test_smoke_failure_removes_module_and_writes_marker(self, tmp_path):
        spec = {"meta": {"slug": "test-slug"},
                "signal": {"definition": "d", "lag_bars": 1}}
        result = {"slug": "test-slug", "source": "x = 1\n"}  # no signal()
        stage = codegen.publish(result, spec, tmp_path / "strategies")
        assert stage == "codegen_failed"
        assert "signal" in (tmp_path / "strategies" / "test-slug.error").read_text()
        # a module that failed smoke is never left at the trusted name, in
        # any form -- no crash can leave the slug marked as coded
        assert not (tmp_path / "strategies" / "test-slug.py").exists()
        assert not (tmp_path / "strategies" / "test-slug.py.tmp").exists()

    def test_superseded_error_marker_removed_on_success(self, tmp_path):
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "test-slug.error").write_text('{"error": "old failure"}\n')
        spec = {"meta": {"slug": "test-slug"},
                "signal": {"definition": "d", "lag_bars": 1}}
        result = {"slug": "test-slug", "source": VALID_MODULE}
        stage = codegen.publish(result, spec, strategies)
        assert stage == "coded"
        assert (strategies / "test-slug.py").exists()
        assert not (strategies / "test-slug.error").exists()

    def test_cli_error_writes_marker_without_module(self, tmp_path):
        result = {"slug": "x", "error": "claude exited 1"}
        stage = codegen.publish(result, {"meta": {"slug": "x"}}, tmp_path / "strategies")
        assert stage == "codegen_failed"
        assert (tmp_path / "strategies" / "x.error").exists()


class TestPendingSpecs:
    def test_skips_coded_slugs(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "a.yaml").write_text("meta:\n  slug: a\n")
        (specs / "b.yaml").write_text("meta:\n  slug: b\n")
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "a.py").write_text("# done\n")
        todo = codegen.pending_specs(specs, strategies, retry=False)
        assert [p.stem for p in todo] == ["b"]

    def test_error_marker_blocks_retry_unless_flag(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "a.yaml").write_text("meta:\n  slug: a\n")
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "a.error").write_text("{}\n")
        assert codegen.pending_specs(specs, strategies, retry=False) == []
        assert [p.stem for p in codegen.pending_specs(specs, strategies, retry=True)] == ["a"]


class TestMain:
    def test_runs_and_resumes_without_articles_jsonl(self, tmp_path, monkeypatch, capsys):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "a.yaml").write_text(
            "meta:\n  slug: a\nsignal:\n  definition: d\n  lag_bars: 1\n")
        strategies = tmp_path / "strategies"

        monkeypatch.setattr(codegen, "ROOT", tmp_path)
        monkeypatch.setattr(codegen, "SPECS", specs)
        monkeypatch.setattr(codegen, "STRATEGIES", strategies)

        calls = []

        def fake_run(*a, **k):
            calls.append(1)

            class Proc:
                returncode = 0
                stdout = VALID_MODULE
                stderr = ""

            return Proc()

        monkeypatch.setattr(codegen.subprocess, "run", fake_run)

        assert codegen.main_with(["--jobs", "1"]) == 0
        assert (strategies / "a.py").exists()
        assert not (tmp_path / "data").exists()  # codegen never touches articles

        # resume: a second run finds nothing to do and makes zero CLI calls
        assert codegen.main_with(["--jobs", "1"]) == 0
        assert len(calls) == 1
