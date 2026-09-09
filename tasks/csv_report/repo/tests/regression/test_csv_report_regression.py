import pytest

from rows import parse_rows
from summary import total


def test_unknown_column_raises_value_error():
    text = """

    name,amount
    Ada,7
    """

    with pytest.raises(ValueError):
        total(text, "missing")


def test_non_integer_body_value_raises_value_error():
    text = """

    name,amount
    Ada,seven
    """

    with pytest.raises(ValueError):
        total(text, "amount")


def test_parse_rows_trims_whitespace_and_preserves_empty_fields():
    text = """

      name , amount 
      Alice , 
    """

    assert parse_rows(text) == [["name", "amount"], ["Alice", ""]]
