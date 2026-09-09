import pytest

from links import extract_links


def test_extract_links_treats_escaped_delimiters_as_literals():
    assert extract_links(r"Ignore \[not a link\] and [te\]xt](u\(rl\)))") == [
        {"text": "te]xt", "url": "u(rl)"},
    ]


def test_extract_links_returns_multiple_links_in_order():
    assert extract_links("One [a](/1) and [b [c]](/2) here.") == [
        {"text": "a", "url": "/1"},
        {"text": "b [c]", "url": "/2"},
    ]


def test_extract_links_returns_empty_list_when_no_links_are_present():
    assert extract_links("Plain text with no inline links.") == []


def test_extract_links_raises_value_error_for_unclosed_link():
    with pytest.raises(ValueError):
        extract_links("Broken [link(/x) with no closing bracket.")
