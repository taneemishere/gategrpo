from slug import slugify


def test_removes_punctuation_and_lowercases():
    assert slugify("Hello, World!") == "hello-world"


def test_collapses_whitespace():
    assert slugify("many   spaces") == "many-spaces"
