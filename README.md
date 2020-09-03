<!--
Template taken from https://github.com/othneildrew/Best-README-Template

*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo, twitter_handle, email
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://litebird-sim.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.com/litebird/litebird_sim.svg?branch=master)](https://travis-ci.com/litebird/litebird_sim)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/litebird/litebird-sim?svg=true)](https://ci.appveyor.com/project/litebird/litebird-sim)
[![Issues][issues-shield]][issues-url]
[![GPL3 License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/litebird/litebird_sim">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">LiteBIRD Simulation Framework</h3>

  <p align="center">
    A set of Python modules to simulate the instruments on board the LiteBIRD spacecraft.
    <br />
    <a href="https://litebird-sim.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/litebird/litebird_sim">View Demo</a>
    ·
    <a href="https://github.com/litebird/litebird_sim/issues">Report Bug</a>
    ·
    <a href="https://github.com/litebird/litebird_sim/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

[![LiteBIRD collaboration][product-screenshot]](https://example.com)



### Built With

-   Love!
-   [Python](https://www.python.org)
-   [Poetry](https://python-poetry.org/)
-   [NumPy](https://numpy.org)
-   [Astropy](https://www.astropy.org)
-   [Healpix](https://healpix.jpl.nasa.gov)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

The LiteBIRD Simulation Framework requires the following tools to be
installed on your system:

1.   Python 3.6 or newer;
2.   [Poetry](https://python-poetry.org/), a Python package to handle
     package dependencies and virtual environments.

You probably already have Python installed on your system. To install
Poetry, use one of the following commands:

-   On Linux/Mac OS X, run the following command from the terminal (no
    `sudo` is required):

    ```
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
    ```

-   On Windows, start a Powershell terminal and run the following command:

    ```
    (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
    ```


### Installation

1.  Install the prerequisites listed above;

2.  Be sure not to have a virtual environment/Conda custom environment
    active, as `poetry` does not plays nicely with them (Poetry handles
    a hidden virtual environment by itself, so there's little point in
    creating one for `litebird_sim`.) See
    [github.com/python-poetry/poetry/issues/1724](https://github.com/python-poetry/poetry/issues/1724)
    for more details.
    
    Run the following commands to clone this repository, install all
    the dependencies and build the framework:

    ```
    git clone git@github.com:litebird/litebird_sim.git
    cd litebird_sim
    poetry install --extras=jupyter --extras=docs
    poetry build
    ```

    There are a few extras you can enable by adding proper `--extras` flag
    to the `poetry install` command above:

    -   `--extras=jupyter` will enable Jupyter;
    -   `--extras=docs` will install Sphinx and all the tools to
        generate the documentation;
    -   `--extras=mpi` will add support for MPI.

3.  Run the set of tests, to check that everything works. On Linux/Mac OS X:

    ```
    ./bin/run_tests.sh
    ```

    On Windows:

    ```
    .\bin\run_tests.bat
    ```


<!-- USAGE EXAMPLES -->
## Usage

The documentation is available online at
[litebird.github.io/litebird_sim/index.html](https://litebird.github.io/litebird_sim/build/html/index.html).

To create a local copy of the documentation, make sure you ran
`poetry` with the flag `--extras=docs`, then run the following
command:

-   Linux or Mac OS X:
    ```
    ./refresh_docs.sh
    ```

-   Windows:
    ```
    poetry shell
    cd docs
    make.bat html
    ```



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/litebird/litebird_sim/issues)
for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

If you are part of the LiteBIRD collaboration and have something that
might fit in this framework, you're encouraged to contact us! Any
contributions you make are **greatly appreciated**.

1.  Read [CONTRIBUTING.md](https://github.com/litebird/litebird_sim/blob/master/CONTRIBUTING.md)
2.  Fork the project
3.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5.  Push to the Branch (`git push origin feature/AmazingFeature`)
6.  Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the [GPL3 License][license-url].


<!-- CONTACT -->
## Contact

LiteBIRD Simulation Team - litebird_pipe@db.ipmu.jp

Project Link: [https://github.com/litebird/litebird_sim](https://github.com/litebird/litebird_sim)



<!-- ACKNOWLEDGEMENTS -->
## How to cite this code

TODO!


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/litebird/litebird_sim?style=flat-square
[issues-url]: https://github.com/litebird/litebird_sim/issues
[license-shield]: https://img.shields.io/github/license/litebird/litebird_sim.svg?style=flat-square
[license-url]: https://github.com/litebird/litebird_sim/blob/master/LICENSE

<!-- Once we have some nice screenshot, let's put a link to it here! -->
[product-screenshot]: images/screenshot.png
