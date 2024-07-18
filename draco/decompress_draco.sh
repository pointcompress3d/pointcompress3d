#!/bin/bash

# Check if two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Assign the input and output directories from command line arguments
input_directory=$1
output_directory=$2

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through all .drc files in the input directory
for file in "$input_directory"/*.drc; do
    # Extract the base filename without the extension
    base_name=$(basename "$file" .drc)

    # Define the output file path with the .ply extension in the output directory
    output="$output_directory/$base_name.ply"

    # Execute the draco_decoder command
    draco_decoder -i "$file" -o "$output"
    
    pcl_ply2pcd -format 0 "$output" "$output_directory/$base_name.pcd"

    rm -rf $output
done
