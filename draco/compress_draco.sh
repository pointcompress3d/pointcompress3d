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

# Array to store paths of temporary .ply files
declare -a temp_ply_files

# Loop through all .pcd files in the input directory
for file in "$input_directory"/*.pcd; do
    # Extract the base filename without the extension
    base_name=$(basename "$file" .pcd)

    # Temporary .ply file path
    temp_ply="$output_directory/$base_name.ply"

    # Convert .pcd to .ply using pcl_pcd2ply
    pcl_pcd2ply -format 0 "$file" "$temp_ply"

    # Add the path of the .ply file to the array
    temp_ply_files+=("$temp_ply")

    # Define the output file path with the new .drc extension in the output directory
    output="$output_directory/$base_name.drc"

    # Execute the draco_encoder command on the converted .ply file
    draco_encoder -qp 3 -qg 8 -point_cloud -i "$temp_ply" -o "$output"
done

# Remove all temporary .ply files after conversion
for ply_file in "${temp_ply_files[@]}"; do
    rm "$ply_file"
done
