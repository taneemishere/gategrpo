import pytest

from config_loader import parse_config


def test_strips_outer_quotes():
    assert parse_config('name = "Ada Lovelace"') == {"name": "Ada Lovelace"}


def test_duplicate_keys_raise_value_error():
    with pytest.raises(ValueError):
        parse_config("host=a\nhost=b")
