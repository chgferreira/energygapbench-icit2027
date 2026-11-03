"""Native installation checks follow declared method selection."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location('selected_native_verifier', ROOT / 'scripts/verify_server_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_verification_defaults_to_full_seven_without_flag(verifier):
    assert verifier.parse_args([]).methods == list(verifier.METHODS)


def test_focused_native_verification_selects_only_its_two_models(verifier):
    args = verifier.parse_args(['--methods', 'SAITS', 'MOMENT-1-base', '--threads', '32'])
    assert args.methods == ['SAITS', 'MOMENT-1-base']
    for methods in [['SAITS', 'SAITS'], ['unknown'], []]:
        with pytest.raises(SystemExit):
            verifier.parse_args(['--methods', *methods])
