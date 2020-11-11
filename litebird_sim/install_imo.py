#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from base64 import b64decode
import getpass
from pathlib import Path
from time import sleep

from github import Github
from rich import print
from rich.table import Table
import tomlkit

CONFIG_PATH = Path.home() / ".config" / "litebird_imo"
CONFIG_FILE_PATH = CONFIG_PATH / "imo.toml"

repositories = []


def get_blob_content(repo, branch, path_name):
    # Taken from https://github.com/PyGithub/PyGithub/issues/661

    # first get the branch reference
    ref = repo.get_git_ref(f"heads/{branch}")
    # then get the tree
    tree = repo.get_git_tree(ref.object.sha, recursive="/" in path_name).tree
    # look for path in tree
    sha = [x.sha for x in tree if x.path == path_name]
    if not sha:
        # well, not found..
        return None
    # we have sha
    return repo.get_git_blob(sha[0])


def retrieve_local_source():
    print()
    path = Path(
        input('Please enter the directory where file "schema.json" resides: ')
    ).absolute()

    if not (path / "schema.json").is_file():
        print(f'[red]Error:[/red] {path} does not seem to contain a "schema.json" file')
        return

    name = input("Now insert a descriptive name for this location: ")

    repositories.append({"name": name, "location": str(path.resolve())})

    print(
        f"""

[green]Repository "{name}" has been added successfully.[/green]

"""
    )


def retrieve_remote_source():
    print(
        """

[red]Sorry, this function is not implemented yet.[/red]

"""
    )


def retrieve_github_source():
    default_user = getpass.getuser()

    user = input(f'Enter your GitHub username (default: "{default_user}"): ').strip()
    if user == "":
        user = default_user

    password = getpass.getpass(prompt="Enter your GitHub password: ")

    g = Github(user, password)

    default_repo = "litebird/litebird_imo"
    repo_name = input(
        f'Enter the GitHub repository (default: "{default_repo}"): '
    ).strip()
    if repo_name == "":
        repo_name = default_repo

    repo = g.get_repo(repo_name)
    branches = list(repo.get_branches())

    # If there are many branches, ask the user to pick one
    if len(branches) > 1:
        print(
            """
Please pick a branch:
"""
        )
        for index, branch in enumerate(branches):
            print(f"{index + 1}.   {branch.name}")

        branch_idx = int(input("\nEnter the number of the branch: ")) - 1
        branch = branches[branch_idx]
    else:
        branch = branches[0]

    # Search for a directory containing file "schema.json"

    locations = []
    print(
        f"""
Using commit [cyan]{branch.commit.sha}[/cyan], modified on
"{branch.commit.stats.last_modified}".
"""
    )
    for cur_file in branch.commit.files:
        if Path(cur_file.filename).name == "schema.json":
            locations.append(cur_file)

    if len(locations) == 0:
        print(
            """

[red]No valid "schema.json" files found in this branch[/red]

"""
        )
        return

    if len(locations) == 1:
        picked_location = locations[0]
    else:
        # Many files are available: ask the user which one to download

        print(
            """

Many files matching the name "schema.json" are present in this commit.
Which one should I use?

"""
        )

        for idx, loc in enumerate(locations):
            print(f"{idx + 1}.   {loc.filename}")

        choice = int(input("Enter the number of the file to use: ")) - 1
        picked_location = locations[choice]

    # Now download the file

    print(
        f"""Downloading file "schema.json" from the following URL:
{picked_location.raw_url}
"""
    )

    schema_file = get_blob_content(repo, branch.name, picked_location.filename)
    assert schema_file.encoding == "base64"
    schema_file = b64decode(schema_file.content)

    # Save the file in a local directory
    imo_local_path = CONFIG_PATH / repo.name / branch.name
    imo_local_path.mkdir(parents=True, exist_ok=True)
    imo_file_path = imo_local_path / "schema.json"
    with imo_file_path.open("wb") as outf:
        outf.write(schema_file)

    print(
        f"""File "schema.json" written into
"{imo_file_path}"
"""
    )

    repositories.append(
        {
            "name": repo.name + "/" + branch.name,
            "location": str(imo_local_path.resolve()),
        }
    )


def run_main_loop() -> bool:
    prompt = """Choose a source for the IMO:

1.   [cyan]The [it]litebird_imo[/it] GitHub repository[/cyan]

     This will download a flatfile representation of the IMO
     and make it available as a "local source" (like point 3).


2.   [cyan]Remote IMO database[/cyan]

     An IMO database running on a separate server.


3.   [cyan]Local source[/cyan]

     A directory on your hard disk.


s.   [cyan]Save and quit[/cyan]

q.   [cyan]Discard modifications and quit[/cyan]

"""

    while True:
        print(prompt)
        choice = input("Pick your choice (1, 2, 3, s or q): ").strip()

        if choice == "1":
            retrieve_github_source()
        elif choice == "2":
            retrieve_remote_source()
        elif choice == "3":
            retrieve_local_source()
        elif choice in ("s", "S"):
            print(
                """

Saving changes and quitting...
"""
            )
            return True

        elif choice in ("q", "Q"):
            print(
                """

Discarding any change and quitting...
"""
            )
            return False

        sleep(2)


def write_toml_configuration():
    file_path = CONFIG_FILE_PATH

    # Create the directory containing the file, if it does not exist.
    # https://github.com/litebird/litebird_sim/issues/61
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wt") as outf:
        outf.write(tomlkit.dumps({"repositories": repositories}))

    print(
        f"""
The configuration has been saved into file
"{str(file_path)}"
"""
    )


def main():
    if run_main_loop():
        write_toml_configuration()
        if len(repositories) > 0:
            print("The following repositories have been configured successfully:")

            table = Table()
            table.add_column("Name")
            table.add_column("Location")

            for row in repositories:
                table.add_row(row["name"], row["location"])

            print(table)

        else:
            print("No repositories have been configured")

    else:
        print("Changes have been discarded")


if __name__ == "__main__":
    main()
