from summary import total


def test_total_uses_header_and_quoted_commas():
    text = """

    name,amount
    "Doe, John",10
    "Roe, Jane",5
    """

    assert total(text, "amount") == 15


def test_total_handles_simple_rows():
    text = """

    name,amount
    Ada,7
    Bob,8
    """

    assert total(text, "amount") == 15
