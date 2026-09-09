from ranges import parse_ranges


def test_expands_simple_ranges():
    assert parse_ranges("1-3,5") == [1, 2, 3, 5]


def test_strips_whitespace():
    assert parse_ranges(" 2 , 4-5 ") == [2, 4, 5]
