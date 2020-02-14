# litebird_sim

Simulation tools for LiteBIRD

## How to install the code

This package uses [poetry](https://python-poetry.org/) to handle
package dependencies. Install it using the following command:

- On Linux/Mac OS X, run the following command from the terminal (no
  `sudo` is required):

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

- On Windows, start a Powershell terminal and run the following command:

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```

Once `poetry` is installed, run the following commands to clone this
repository, install all the dependencies and build a package:

```
git clone git@github.com:litebird/litebird_sim.git
cd litebird_sim
poetry install --extras=jupyter
poetry build
```

There are a few extras you can enable by adding proper `--extras` flag
to the `poetry install` command above:

- `--extras=jupyter` will enable Jupyter;
- `--extras=docs` will install Sphinx and all the tools to generate
  the documentation;
- `--extras=mpi` will add support for MPI.


## How to run tests

You can use [pytest](https://docs.pytest.org/en/latest/), but remember
to run `poetry`:

```
poetry run python -m pytest
```
