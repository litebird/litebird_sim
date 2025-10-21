#!/usr/bin/env python3

import requests
from typing import Any


def clean_tag_name(tag: str) -> str:
    return tag.removeprefix("v")


REL_URL = "https://api.github.com/repos/litebird/litebird_sim/releases"


def print_email_text(release: dict[str, Any]):
    print(
        """
Dear all,

   We are happy to announce that version {tag_name} of the LiteBIRD Simulation
Framework has been released. You can get it from PyPI using the command

    pip install --upgrade litebird_sim

or directly from the URL

    https://github.com/litebird/litebird_sim/releases

Here is a list of the changes:

{body}

Best,
  The LiteBIRD Simulation Team.
    """.format(tag_name=clean_tag_name(release["tag_name"]), body=release["body"])
    )


def main():
    resp = requests.get(REL_URL)

    releases = resp.json()
    last_release = releases[0]

    print_email_text(last_release)


if __name__ == "__main__":
    main()
