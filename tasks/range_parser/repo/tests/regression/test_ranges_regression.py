import pytest

from ranges import parse_ranges


def test_removes_duplicates_preserving_order():
    assert parse_ranges("1,1,2-3,2") == [1, 2, 3]


def test_descending_range_raises_value_error():
    with pytest.raises(ValueError):
        parse_ranges("4-2")
