# Contributing to litebird_sim

This document provides information about how to contribute to the
development of `litebird_sim`. If you are reading it, chances are that
you are willing to help us: thanks!

This repository is hosted at https://github.com/litebird/litebird_sim,
and all the development should happen there. Be sure to check that you
have access to the repository, as it is private; if you cannot open
the URL, please ask either Maurizio Tomasi <maurizio.tomasi AT
unimi.it> or Davide Poletti <davide.poletti AT sissa.it> to grant you
access. (Note that you must already have a GitHub account, as you have
to provide your username to them.)


## Setting up a developer's environment

We try to support Windows machines, but the developer's environment we
support is based on Linux or Mac OS X. Windows users should install
[Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (we
suggest the Ubuntu flavour) and make all the development from a *WSL*
terminal. (Note that WSL is not supported on Windows 7 or 8; in this
case, better to install a virtual machine running Ubuntu using
[Virtual Box](https://www.virtualbox.org/).)

If you are running a Linux/Mac OS X machine, you must first ensure you
have the following tools:

- A Python 3 implementation. You can use the bare
  [CPython](https://www.python.org/) implementation, which is most
  likely already installed on your system, or
  [Miniconda](https://conda.io/en/latest/miniconda.html).
- [Git](https://git-scm.com/). If you have never used Git, it's better
  to read some book or ask a colleague. You should be familiar with
  concepts such as *commits*, *branches*, and *pull requests* at
  least.

When you have installed those dependencies, run the following commands
to install [uv](https://docs.astral.sh/uv/) and set up a number of
Git hooks:

```sh
# Install uv
pip install uv

# Clone the litebird_sim repository
git clone git@github.com:litebird/litebird_sim.git
cd litebird_sim

# Install the dependencies and create a virtual environment.
# You can skip some --extras if you do not need them, but
# be sure to install "docs"
uv sync --all-extras
uv build

# Install Git hooks
sh ./bin/git/init-hooks
```

The latter command will install a `pre-commit` Git hook that will be
run whenever you execute `git commit`. The script will check the
quality of your commit, using the following tools:

- [Ruff](https://github.com/astral-sh/ruff) checks for code formatting
  issues, common Python errors, and stylistic issues (unused imports, etc.)
- [Pytest](https://docs.pytest.org/en/latest/) run automatic tests on
  the code.

In order to maintain the code readable to everybody in the
collaboration, you must run `ruff format` on your source code before
committing anything, otherwise the commit will be rejected. You can
configure your editor to automatically use `ruff` to reformat the
code, or you can run the following command before issuing a `git
commit`:

```sh
cd litebird_sim
ruff format .
```

It's a good practice to run tests on the code while you are developing
a feature, even if you are not going to commit them soon. Use the
following script to run all the checks:

```sh
cd litebird_sim
./bin/run_tests.sh
```


## Using Pull Requests to work on the code

You should *never* commit on the `master` branch of the
repository. Instead, create a dedicated branch and name it properly:

```
git checkout -b my_awesome_feature
```

Do all the work and commit whatever you want, remembering to run
`./bin/run_tests.sh` periodically to check that your changes complies
with the coding standards used in the `litebird_sim` framework.

When you are completed, run the following command:

```
git push --set-upstream origin my_awesome_feature
```

This will push your modifications to the remote repository hosted on
GitHub and write some information on the screen. The most important is
an URL in the form
`https://github.com/litebird/litebird_sim/pull/NNN`, with `NNN` being
a number: follow the link and enter a few details about your
change. Press the green button to submit a *pull request*, which will
be reviewed by the Simulation Team. If everything goes well, your
changes will be incorporated in `litebird_sim`, and
[CHANGELOG.md](https://github.com/litebird/litebird_sim/blob/master/CHANGELOG.md)
will be updated.


## Practical tips when implementing a change

When you are working on a feature, you must ensure that your
modifications fulfill the following goals:

-   The code should be clear and pleasing to read:

    -   No strange variable names like `ll`, `xwv`
    
    -   Measure units should be indicated in variables, parameters and
        keywords when appropriate (e.g., use `temperature_k` instead
        of just `temperature`)

    -   [No](https://github.com/godotengine/godot/commit/d35e48622800f6686dbdfba380e25170005dcc2b)
        [bad](https://www.zdnet.com/article/linux-patch-replaces-f-words-with-hugs-in-kernel-comments-but-some-cry-censorship/)
        [words](https://softwareengineering.stackexchange.com/questions/50928/dealing-with-profanity-in-source-code)
        [in](https://www.quora.com/How-common-is-it-for-programmers-to-use-profanity-in-variable-names)
        [your](https://www.sitepoint.com/community/t/php-code-to-replace-bad-words-in-a-document-with-and-maintain-break-line/286147)
        [code](https://pypi.org/project/profanity-check/)!

-   Every functionality should be documented, both with docstrings and
    with text in proper places in the documentation (see the `docs`
    folder). Once you update the documentation, remember to run the
    script (Linux, Mac OS X)
    
    ```
    ./bin/refresh_docs.sh
    ```
    
    Under Windows, run the following command:
    
    ```
    uv run docs\make.bat html
    ```
    
-   Provide unit tests and possibly integration tests, adding scripts
    in the `test` directory. Name your files `test_*.py` and be sure
    they are run by `pytest`.
