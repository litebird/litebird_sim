#!/bin/sh

# Re-generate the Sphinx documentation
poetry run make -C docs html
