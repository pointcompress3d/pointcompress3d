import cv2
import numpy as np


base_path = '/data/temp/'

def fill_nan_with_previous(image):
    # Check if the image is single-channel
    if len(image.shape) != 2:
        raise ValueError("Image must be a single-channel grayscale image.")

    # Iterate over each row in the image
    for i in range(image.shape[0]):
        # Iterate over each column in the row
        for j in range(1, image.shape[1]):
            # If the current pixel is NaN, replace it with the previous pixel value
            if image[i, j] == 0 and j-1 >= 0:
                image[i, j] = image[i, j-1]


def loadImage(type='r', index=0):
    if type == 'r':
        pass
    else if type == 'a':
        pass
    else if type == 'i':
        pass


index = 3
_rangeImage = loadImage('r', index) 
_intensityMap = loadImage('a', index)
_azimuthMap = loadImage('i', index)

cols = 2048
rows = 64
_max_range = 150
_min_range = 0


#rangeImageSph_

factor = 1.0/(_max_range - _min_range)
offset = - _min_range
range_image_path = '/data/temp/s110_lidar_ouster_south_range/1698077878_072571902.jpg'
azi_image_path = '/'

output_path = '/data/temp/s110_lidar_ouster_south_range/output.jpg'
range_image = cv2.imread(range_image_path, cv2.IMREAD_ANYDEPTH)
print(range_image.shape)


#fill_nan_with_previous(range_image)
#cv2.imwrite(output_path, range_image)
