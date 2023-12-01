# How to prepare a new release

-   Update the version number in the following files:

    -   `pyproject.toml`

    -   `litebird_sim/version.py`

    -   `docs/source/conf.py`

    -   `CHANGELOG.md` (be sure to leave an empty `HEAD` title at the
        top);

    Commit all the changes to `master`.
    
-   Build the release:

    ```
    poetry build
    ```

-   Upload the package to PyPI:

    ```
    poetry upload
    ```

-   Create a new tag and push it to GitHub:

    ```
    git tag -a vX.Y.Z -m "Version X.Y.Z"
    git push origin vX.Y.Z
    ```
    
-   Open the page https://github.com/litebird/litebird_sim/releases and create a new release from the tag you have just created.

-   Use the script `prepare_release_email.py` to automatically produce the text of an email containing the release notes:

    ```
    $ python3 prepare_release_email.py | xclip -selection clipboard
    ```
    
    and send this text to the LiteBIRD JSGs.

    
