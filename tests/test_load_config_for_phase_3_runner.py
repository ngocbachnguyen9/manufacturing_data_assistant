import os
import yaml
import pytest
import importlib.util

# adjust this to point at the file that defines load_config()
SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "run_phase3_human_study.py")
)

@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_load_config_success(tmp_path, runner):
    cfg = {"foo": 123, "bar": ["x", "y"]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))

    loaded = runner.load_config(str(p))
    assert loaded == cfg

def test_load_config_missing(runner):
    with pytest.raises(FileNotFoundError):
        runner.load_config("no_such_file.yaml")