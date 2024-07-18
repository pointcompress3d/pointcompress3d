import os
import glob
import numpy as np
import argparse
from open3d import *


def convert_pcd_to_kitti(pcdFilePath):
    pcd = io.read_point_cloud(pcdFilePath)
    points = np.asarray(pcd.points)
    points = points.T
    return points.astype('float32')#.tofile(pcdFilePath+'.bin')

def convert_all_pcd_to_kitti(directory_path):
    # Get a list of all PCD files in the specified directory
    pcd_files = glob.glob(os.path.join(directory_path, '*.pcd'))

    # Iterate over each PCD file and convert to KITTI format
    for pcd_file in pcd_files:
        convert_pcd_to_kitti(pcd_file)

def main():
    parser = argparse.ArgumentParser(description="Point cloud converter from PCD to KITTI format")
    
    parser.add_argument('--input', type=str, help='First number', required=True)
    parser.add_argument('--output_dir', type=str, help='Second number', required=False)

    args = parser.parse_args()

    converted = convert_pcd_to_kitti(args.input)
    output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.input))
    filename = os.path.basename(args.input).split(".")[0]
    converted.tofile(f"{output_dir}/{filename}.pcd.bin")

if __name__ == "__main__":
    main()
