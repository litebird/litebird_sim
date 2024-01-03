set /a MAX_LINE_LENGTH = 88

rem Verify that the code is properly formatted
poetry run ruff format --diff .

rem Check for common errors
poetry run ruff check .

rem Run the test suite
poetry run python3 -m pytest --doctest-modules -vv
