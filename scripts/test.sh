#!/bin/bash

echo "Running serial tests..."
pytest -m "serial" -n 0

echo "Running parallel tests..."
pytest -m "not serial" -n auto