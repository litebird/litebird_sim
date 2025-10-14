set /a MAX_LINE_LENGTH = 88

rem Run the test suite
uv run python -m pytest --doctest-modules -vv
