# How to prepare a new release

-   Update the version number in the following files:

    -   `pyproject.toml`

    -   `litebird_sim/version.py`

    -   `docs/source/conf.py`

    -   `CHANGELOG.md` (be sure to leave an empty `HEAD` title at the
        top);
    
-   Build the release:

    ```
    poetry build
    ```

-   Upload the `.tar.gz` and `.whl` files to the PyPI Test server:

    ```
    twine upload --repository-url https://test.pypi.org/legacy/ dist/litebird_sim_*
    ```

-   Check that everything looks right by opening the URL shown by Twine

-   If everything looks ok, upload the package to PyPI:

    ```
    twine upload dist/litebird_sim_*
    ```
