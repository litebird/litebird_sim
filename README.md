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


## Source code formatting

The code must be formatted using
[`black`](https://github.com/psf/black).


## How to run tests

Use the following script to run tests on the code:

```
./bin/run_test.sh
```

On Windows, run this command:

```
.\bin\run_test.bat
```


## How to contribute

See file
[CONTRIBUTING.md](https://github.com/litebird/litebird_sim/blob/master/CONTRIBUTING.md)
