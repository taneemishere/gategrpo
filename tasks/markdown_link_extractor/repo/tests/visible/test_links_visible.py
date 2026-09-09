from links import extract_links


def test_extract_links_handles_basic_inline_link():
    assert extract_links("Read [docs](https://example.com).") == [
        {"text": "docs", "url": "https://example.com"},
    ]


def test_extract_links_handles_parentheses_in_url():
    assert extract_links("Read [wiki](https://en.wikipedia.org/wiki/Foo_(bar_(baz))) now.") == [
        {"text": "wiki", "url": "https://en.wikipedia.org/wiki/Foo_(bar_(baz))"},
    ]


def test_extract_links_handles_nested_brackets_in_link_text():
    assert extract_links("See [see [ref] here](/x) for details.") == [
        {"text": "see [ref] here", "url": "/x"},
    ]
