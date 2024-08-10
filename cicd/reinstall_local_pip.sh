#!/bin/bash
rm -Rf dist/*
pip uninstall lemonpepper
python -m build
pip install .