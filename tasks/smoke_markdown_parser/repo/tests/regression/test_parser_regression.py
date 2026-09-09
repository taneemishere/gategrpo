import pytest

from parser import parse_table


def test_empty_cells_are_preserved():
    markdown = """
    | name | value |
    | --- | --- |
    | alpha | |
    """

    assert parse_table(markdown) == [{"name": "alpha", "value": ""}]


def test_malformed_row_raises_value_error():
    markdown = """
    | name | value |
    | --- | --- |
    | alpha | 1 | extra |
    """

    with pytest.raises(ValueError):
        parse_table(markdown)
