from parser import parse_table


def test_parses_basic_markdown_table():
    markdown = """
    | name | value |
    | --- | --- |
    | alpha | 1 |
    | beta | 2 |
    """

    assert parse_table(markdown) == [{"name": "alpha", "value": "1"}, {"name": "beta", "value": "2"}]


def test_escaped_pipe_stays_inside_cell():
    markdown = r"""
    | name | value |
    | --- | --- |
    | alpha | one \| two |
    """

    assert parse_table(markdown) == [{"name": "alpha", "value": "one | two"}]
