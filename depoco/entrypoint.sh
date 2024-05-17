#!/bin/bash

echo "Converting point cloud"

INPUT_PATH=$1
OUTPUT_PATH=$2

if [ "$OUTPUT_PATH" ]; then
    python3 converter.py --input "$INPUT_PATH" --output_dir "$OUTPUT_PATH"
else
    python3 converter.py --input "$INPUT_PATH"
fi
