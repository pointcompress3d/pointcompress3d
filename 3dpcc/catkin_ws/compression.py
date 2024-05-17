import cv2
import os
import sys

# Directory containing input PNG images
input_dir = sys.argv[1] #'input_images/'

# Directory to save compressed JPEG images
output_dir = sys.argv[2] #'compressed_images/'

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
files = os.listdir(input_dir)

# Iterate over each file in the directory
for file in files:
    if file.endswith('.png'):
        # Load the PNG image
        image = cv2.imread(os.path.join(input_dir, file))

        if image is None:
            print(f"Could not open or find the image: {file}")
        else:
            # Define compression parameters
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # JPEG compression quality (0-100), higher means better quality but larger file size

            # Compress the image using JPEG compression
            success, compressed_image = cv2.imencode('.jpg', image, compression_params)

            if not success:
                print(f"Failed to compress the image: {file}")
            else:
                # Decode the compressed image
                compressed_image = cv2.imdecode(compressed_image, 1)

                # Save the compressed image
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.jpg')
                cv2.imwrite(output_path, compressed_image)
                print(f"Image compression complete: {output_path}")

