# How to prepare a new release

Building, uploading to PyPI and generating the release email are handled
automatically by the `Upload to PyPI` workflow (`.github/workflows/publish.yml`),
which runs whenever a GitHub Release is published. The manual steps are therefore
just the version bump and the creation of the Release itself.

-   Update the version number. The version lives in a single place,
    `litebird_sim/version.py` (`__version__`); `pyproject.toml` and
    `docs/source/conf.py` read it from there, so this is the only file to edit.

-   Update `CHANGELOG.md`: rename the top `HEAD` heading to the new version
    number and leave a fresh empty `HEAD` heading above it.

    Commit all the changes to `master`.

-   If the supported versions for Python changed, be sure to change
    the chapter “Installation” in the documentation.

-   On GitHub, open <https://github.com/litebird/litebird_sim/releases> and
    draft a new Release. Create a new tag `vX.Y.Z` targeting `master`, paste the
    relevant `CHANGELOG.md` section as the release notes, and publish it.

-   Publishing the Release triggers the `Upload to PyPI` workflow, which builds
    the package and uploads it to PyPI via trusted publishing (no token needed).
    Check the workflow run to confirm it succeeded.

-   The same workflow runs `bin/prepare_release_email.py` and uploads the result
    as a `release-email` artifact. Download it from the workflow run and send its
    contents to the LiteBIRD JSGs.
