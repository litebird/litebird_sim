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

-   Create a new tag and push it to GitHub:

    ```
    git tag -a vX.Y.Z -m "Version X.Y.Z"
    git push origin vX.Y.Z
    ```
    
-   Open the page https://github.com/litebird/litebird_sim/releases and create a new release from the tag you have just created.
