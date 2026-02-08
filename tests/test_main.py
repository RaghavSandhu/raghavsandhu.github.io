from src.main import greet


def test_greet():
    assert greet("world") == "Hello, world!"


def test_greet_custom_name():
    assert greet("Raghav") == "Hello, Raghav!"
