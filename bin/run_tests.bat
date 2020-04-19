set /a MAX_LINE_LENGTH = 88

rem Verify that the code is properly formatted
poetry run black --check --line-length %MAX_LINE_LENGTH% -q .

rem Check for common errors
poetry run flake8 --max-line-length %MAX_LINE_LENGTH%

rem Run the test suite
poetry run python3 -m pytest --doctest-modules -vv
