from stats import average


def test_average_of_empty_list():
    assert average([]) == 0.0
