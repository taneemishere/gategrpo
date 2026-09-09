from config_loader import parse_config


def test_ignores_comments_and_blanks():
    text = """
    # service config
    host = example.com

    port = 443
    """

    assert parse_config(text) == {"host": "example.com", "port": "443"}


def test_value_may_contain_equals():
    assert parse_config("token = a=b=c") == {"token": "a=b=c"}
