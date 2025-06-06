# runtests.py
import sys, pathlib
# ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest

sys.exit(pytest.main(sys.argv[1:]))