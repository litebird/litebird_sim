set /a MAX_LINE_LENGTH = 88

rem Verify that the code is properly formatted
uv run ruff format --diff .

rem Check for common errors
uv run ruff check .

rem Run the test suite
uv run python -m pytest --doctest-modules -vv
