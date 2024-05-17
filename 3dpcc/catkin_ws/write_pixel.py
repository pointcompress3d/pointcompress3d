import cv2
import numpy as np
import sys

# Load the image
image_path = sys.argv[1]
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Ensure the image is mono16 (16-bit grayscale)
if image.dtype != 'uint16':
    raise ValueError("Image is not in mono16 format.")

# Open a file for writing
output_file = sys.argv[2]
with open(output_file, 'w') as f:
    # Iterate over rows and columns of the matrix
    for row in image:
        for pixel_value in row:
            # Write each pixel value to the file
            f.write(str(pixel_value) + " ")
        f.write("\n")

print("Matrix written to", output_file)

