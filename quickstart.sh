#!/bin/bash
# Convenience wrapper for data preparation quickstart

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$SCRIPT_DIR/scripts/data_preparation/quickstart.sh" "$@"
