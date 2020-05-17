#!/bin/bash

set -o nounset
set -o errexit

readonly LITEBIRD_SIM_PATH="$HOME/litebird_sim"

# Update APT database and install basic Python3 tools
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-distutils python3-pip git

python3 --version

# Install Poetry
sudo pip install poetry

# Useful for debugging
poetry debug

# Install the litebird_sim repository and test it
git clone https://github.com/litebird/litebird_sim.git "$LITEBIRD_SIM_PATH"
cd "$LITEBIRD_SIM_PATH"
poetry install --extras=docs
