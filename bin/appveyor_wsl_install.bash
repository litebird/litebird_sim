#!/bin/bash

set -o nounset
set -o errexit

readonly POETRY_PATH="$HOME/.poetry/bin/poetry"
readonly LITEBIRD_SIM_PATH="$HOME/litebird_sim"

# Update APT database and install basic Python3 tools
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-distutils git

python3 --version

# Install Poetry
curl -sSL -o "$HOME/get-poetry.py" https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
python3 "$HOME/get-poetry.py" -f -y

# Useful for debugging
python3 "$POETRY_PATH" debug

# Install the litebird_sim repository and test it
git clone https://github.com/litebird/litebird_sim.git "$LITEBIRD_SIM_PATH"
cd "$LITEBIRD_SIM_PATH"
python3 "$POETRY_PATH" install --extras=docs

