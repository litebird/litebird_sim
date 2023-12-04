#!/bin/sh

set -o nounset
set -o errexit

readonly LITEBIRD_SIM_PATH="$HOME/litebird_sim"

echo "# Entering $LITEBIRD_SIM_PATH"
cd "$LITEBIRD_SIM_PATH"

echo "# Running tests"
bash bin/run_tests.sh
