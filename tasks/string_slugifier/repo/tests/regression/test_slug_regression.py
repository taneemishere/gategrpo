from slug import slugify


def test_collapses_existing_hyphen_runs():
    assert slugify("Already--Slug") == "already-slug"


def test_symbol_only_input_returns_empty_string():
    assert slugify(" !!! ") == ""
